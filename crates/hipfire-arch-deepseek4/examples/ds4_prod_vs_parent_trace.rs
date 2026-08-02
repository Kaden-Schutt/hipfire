// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Production DS4 vs parent residual-norm trajectory.
//!
//! The parent forward scores PPL ~164 on the Gate-6 corpus while the
//! production MQ2R path scores ~14.7 on the same tokens. Shared f64 oracles
//! cannot catch a misreading baked into both GPU and oracle. This binary uses
//! the production path as the independent oracle and dumps the cheapest
//! diagnostic: **per-layer HC residual L2** (and consecutive-layer ratios).
//!
//! Modes:
//! - `--mode prod`  — load MQ2R HFQ, sequential `decode_step` with
//!   `HIPFIRE_DEEPSEEK4_LAYER_NORM=1`, print residual L2 after every layer at
//!   the last decoded position.
//! - `--mode parent` — load the parent checkpoint, run
//!   `parent_model_forward_traced`, print the multi-row residual L2 series.
//! - `--mode compare` — print the pinned parent baseline series (no GPU) next
//!   to a previously-captured prod CSV, plus the structural checklist.
//!
//! ```text
//! HIPFIRE_DEEPSEEK4_LAYER_NORM=1 cargo run --release -p hipfire-arch-deepseek4 \
//!   --example ds4_prod_vs_parent_trace -- \
//!   --mode prod \
//!   --model /mnt/scratch/quantization/deepseek-v4-flash-0731-mq2r-p3/artifacts/deepseek-v4-flash-0731.mq2r \
//!   --expect-sha256 cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce \
//!   --token-ids /mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin \
//!   --rows 128 --csv /tmp/prod_layer_norms.csv
//!
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_prod_vs_parent_trace -- \
//!   --mode parent \
//!   --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
//!   --token-ids /mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin \
//!   --rows 128 --csv /tmp/parent_layer_norms.csv
//!
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_prod_vs_parent_trace -- \
//!   --mode compare --prod-csv /tmp/prod_layer_norms.csv
//! ```
//!
//! Must run prod/parent modes on gfx942 (mi300x).

use hipfire_arch_deepseek4::forward::{decode_step, take_layer_norm_trace};
use hipfire_arch_deepseek4::parent::forward::{
    parent_layer_forward, parent_layer_forward_traced, ParentForwardScratch, ParentLayerTrace,
    PARENT_HC_DIM,
};
use hipfire_arch_deepseek4::parent::head::{parent_embed, PARENT_VOCAB};
use hipfire_arch_deepseek4::parent::inventory::ParentInventory;
use hipfire_arch_deepseek4::parent::manifest::sha256_file;
use hipfire_arch_deepseek4::parent::model::{
    parent_model_forward_traced, LayerHcNormStats, ParentModelScratch,
};
use hipfire_arch_deepseek4::parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_arch_deepseek4::parent::Ds4ParentBackend;
use hipfire_arch_deepseek4::DeepseekV4;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_PARENT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const DEFAULT_MQ2R: &str =
    "/mnt/scratch/quantization/deepseek-v4-flash-0731-mq2r-p3/artifacts/deepseek-v4-flash-0731.mq2r";
const DEFAULT_MQ2R_SHA: &str =
    "cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce";
const DEFAULT_TOKEN_IDS: &str =
    "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin";
const DEFAULT_ROWS: usize = 128;

/// Pinned parent 1024-token residual L2 from Gate-6 `parent_1024_promoted.log`
/// (engine commit f8b98f0a2, logits sha 65e8e75b…). Multi-row L2 over all 1024
/// positions × 4 streams × 4096.
const PARENT_BASELINE_1024: &[f64] = &[
    494.179871, 474.714539, 483.457733, 482.975098, 486.401825, 777.972900,
    1188.696289, 1263.666992, 1483.049683, 1808.714600, 2081.460205, 2448.574463,
    2984.153564, 3357.408936, 3460.070312, 3531.552002, 3701.159180, 4005.140137,
    4563.366699, 4789.596191, 5978.350586, 6502.970703, 7430.394531, 7603.702148,
    9409.650391, 12910.570312, 41513.074219, 52746.164062, 63270.675781, 67993.132812,
    89999.773438, 127263.906250, 157817.453125, 189329.296875, 274753.968750,
    364817.343750, 426390.687500, 510029.375000, 618344.125000, 643934.000000,
    677608.437500, 670448.625000, 631609.125000,
];

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
    match args.mode.as_str() {
        "prod" => run_prod(&args),
        "parent" => run_parent(&args),
        "compare" => run_compare(&args),
        other => Err(format!("unknown --mode {other} (want prod|parent|compare)")),
    }
}

// ── prod ────────────────────────────────────────────────────────────────────

fn run_prod(args: &Args) -> Result<(), String> {
    // SAFETY: single-threaded main before any concurrent readers.
    unsafe {
        std::env::set_var("HIPFIRE_DEEPSEEK4_LAYER_NORM", "1");
        // Hand loop so the dump sits next to hc_ffn_mix without lowered
        // binding indirection; byte-identical on hipx to the lowered path.
        std::env::set_var("HIPFIRE_FORWARD_LOWERED", "0");
    }

    let model_path = Path::new(&args.model);
    if !model_path.is_file() {
        return Err(format!(
            "--model must be an HFQ file, got {}",
            model_path.display()
        ));
    }
    let expect = args.expect_sha256.as_deref().unwrap_or(DEFAULT_MQ2R_SHA);
    println!("=== ds4_prod_vs_parent_trace (prod) ===");
    println!("model: {}", model_path.display());
    println!("expect-sha256: {expect}");
    println!("token-ids: {}", args.token_ids.display());
    println!("rows: {}", args.rows);

    if args.skip_sha256 {
        println!("sha256: SKIPPED (--skip-sha256; pinned expect={expect})");
    } else {
        let model_sha = sha256_file(model_path)?;
        if !eq_hex_ci(&model_sha, expect) {
            return Err(format!(
                "model sha256 mismatch (got {model_sha}, want {expect})"
            ));
        }
        println!("sha256: OK ({model_sha})");
    }

    let token_ids = read_token_ids(&args.token_ids)?;
    let n = args.rows.min(token_ids.len());
    if n == 0 {
        return Err("token-ids empty".into());
    }
    let token_ids = &token_ids[..n];
    println!("n_tokens: {n}");

    let mut hfq = HfqFile::open(model_path).map_err(|e| format!("open HFQ: {e:?}"))?;
    let mut cfg = DeepseekV4::config_from_hfq(&hfq)?;
    cfg.load_dspark = false;
    println!(
        "config: layers={} hidden={} hc_mult={} mq2r={} route_scale_cfg={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.hc_mult,
        cfg.mq2r,
        cfg.routed_scaling_factor
    );
    println!(
        "note: production mhc_pre default post_scale=1.5 (env HIPFIRE_DEEPSEEK4_POST_SCALE); \
         parent/reference uses 2*sigmoid (hardcoded 2.0). Do NOT import 1.5 into parent."
    );

    let mut gpu = Gpu::init().map_err(|e| format!("Gpu::init: {e:?}"))?;
    println!("gpu: {}", gpu.arch);
    if !gpu.arch.contains("gfx942") && std::env::var_os("HIPFIRE_DS4_ALLOW_NON_GFX942").is_none() {
        return Err(format!(
            "gfx942 required (got {}); set HIPFIRE_DS4_ALLOW_NON_GFX942=1 to override",
            gpu.arch
        ));
    }

    let load_t0 = Instant::now();
    let mut state = DeepseekV4::new_state(&mut gpu, &cfg)?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    drop(hfq);
    println!("loaded in {:.3} s", load_t0.elapsed().as_secs_f64());

    // Sequential decode so SWA/compressor see a real window. Capture the
    // residual series at the LAST position (full context).
    println!("=== forward (sequential decode_step 0..{n}) ===");
    println!("PROD_LAYER_NORM lines also go to stderr.");
    let fwd_t0 = Instant::now();
    let mut last_logits_l2 = 0.0f64;
    for (pos, &tok) in token_ids.iter().enumerate() {
        let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos as u32)
            .map_err(|e| format!("decode_step pos={pos}: {e}"))?;
        if pos + 1 == n {
            last_logits_l2 = l2_f64(&logits);
        }
        if pos % 32 == 0 || pos + 1 == n {
            eprintln!(
                "prod progress pos={pos}/{} elapsed={:.1}s",
                n - 1,
                fwd_t0.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "forward done in {:.3} s; last-row logits L2 = {last_logits_l2:.6}",
        fwd_t0.elapsed().as_secs_f64()
    );

    let (pos, series) = take_layer_norm_trace().ok_or_else(|| {
        "no PROD_LAYER_NORM series captured — is HIPFIRE_DEEPSEEK4_LAYER_NORM wired?".to_string()
    })?;
    println!();
    println!(
        "=== production per-layer residual L2 (single-token, pos={pos}, nelems=hc_mult*hidden) ==="
    );
    print_trajectory("prod", &series);

    if let Some(csv) = args.csv.as_ref() {
        write_csv_series(csv, &series)?;
        println!("wrote {}", csv.display());
    }

    print_structural_diff();
    print_embed_head_boundary();
    Ok(())
}

// ── parent ──────────────────────────────────────────────────────────────────

fn run_parent(args: &Args) -> Result<(), String> {
    let model_path = Path::new(&args.model);
    if !model_path.is_dir() {
        return Err(format!(
            "--model must be a safetensors directory, got {}",
            model_path.display()
        ));
    }
    println!("=== ds4_prod_vs_parent_trace (parent) ===");
    println!("model: {}", model_path.display());
    println!("token-ids: {}", args.token_ids.display());
    println!("rows: {}", args.rows);

    let token_ids = read_token_ids(&args.token_ids)?;
    let n = args.rows.min(token_ids.len());
    if n == 0 {
        return Err("token-ids empty".into());
    }
    let token_ids = &token_ids[..n];

    let source = SafetensorsSource::open(model_path).map_err(|e| {
        format!(
            "SafetensorsSource::open({}): {e}",
            model_path.display()
        )
    })?;

    let mut gpu = Gpu::init().map_err(|e| format!("Gpu::init: {e:?}"))?;
    println!("gpu: {}", gpu.arch);
    if gpu.try_gfx942().is_none() && std::env::var_os("HIPFIRE_DS4_ALLOW_NON_GFX942").is_none() {
        return Err(format!(
            "gfx942 required (got {}); set HIPFIRE_DS4_ALLOW_NON_GFX942=1 to override",
            gpu.arch
        ));
    }

    let (backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    println!(
        "admit OK: layers={} hash_layers={} n_routed={} topk={}",
        cfg.num_hidden_layers, cfg.num_hash_layers, cfg.n_routed_experts, cfg.num_experts_per_tok
    );

    let inv = ParentInventory::build(&source, &cfg)?;
    let plan = ParentLoadPlan {
        layers: 0..cfg.num_hidden_layers,
        load_experts: true,
    };
    let load_t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    println!("loaded in {:.3} s", load_t0.elapsed().as_secs_f64());

    let mut layer_norms: Vec<LayerHcNormStats> = Vec::new();
    let mut compress_events: Vec<(usize, usize)> = Vec::new();
    let logits = zeros_f32(&mut gpu, &[n, PARENT_VOCAB])?;
    let mut scratch = ParentModelScratch::new(&mut gpu, &cfg, n)?;
    let fwd_t0 = Instant::now();
    parent_model_forward_traced(
        &mut gpu,
        backend,
        &weights,
        &cfg,
        &mut scratch,
        token_ids,
        /*start_pos=*/ 0,
        &logits,
        &mut layer_norms,
        &mut compress_events,
    )?;
    println!("forward done in {:.3} s", fwd_t0.elapsed().as_secs_f64());

    // Median per-row L2, not the aggregate: one massive-activation row (row 0
    // at L37→L38: median 404→413 against aggregate 14222→116670) dominates the
    // flat L2 and makes a healthy stack look like it is diverging.
    let series: Vec<f64> = layer_norms.iter().map(|s| s.median as f64).collect();
    println!();
    println!("=== parent per-layer HC residual, MEDIAN per-row L2 ({n} rows) ===");
    print_trajectory("parent", &series);

    if let Some(csv) = args.csv.as_ref() {
        write_csv_series(csv, &series)?;
        println!("wrote {}", csv.display());
    }

    if args.stage_dump {
        dump_parent_stages(&mut gpu, backend, &weights, &cfg, token_ids, n, &args.stage_layers)?;
    }

    print_structural_diff();
    print_embed_head_boundary();
    let _ = compress_events;
    Ok(())
}

fn dump_parent_stages(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    weights: &ParentWeights,
    cfg: &hipfire_arch_deepseek4::parent::ParentQuantConfig,
    token_ids: &[u32],
    n: usize,
    stage_layers: &[usize],
) -> Result<(), String> {
    println!();
    println!(
        "=== parent per-stage L2 at layers {:?} (rows={n}) ===",
        stage_layers
    );
    let mut layer_scratch = ParentForwardScratch::new(gpu, cfg, n)?;
    // HC ping-pong buffers.
    let hc_a = zeros_f32(gpu, &[n, PARENT_HC_DIM])?;
    let hc_b = zeros_f32(gpu, &[n, PARENT_HC_DIM])?;
    // Need KV rings: allocate one ring per layer we will run (0..=max stage).
    let max_l = *stage_layers.iter().max().unwrap_or(&0);
    let end = (max_l + 1).min(cfg.num_hidden_layers);
    // SWA ring shape from parent attention constants.
    use hipfire_arch_deepseek4::parent::attention::{
        PARENT_HEAD_DIM, PARENT_N_KV_HEADS, PARENT_SWA_WINDOW,
    };
    let mut rings: Vec<GpuTensor> = Vec::with_capacity(end);
    for _ in 0..end {
        rings.push(zeros_f32(
            gpu,
            &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
        )?);
    }
    parent_embed(gpu, backend, weights, cfg, token_ids, &hc_a)?;
    let mut use_a = true;
    for layer_idx in 0..end {
        let (x, out) = if use_a {
            (&hc_a, &hc_b)
        } else {
            (&hc_b, &hc_a)
        };
        let input_ids = if layer_idx < cfg.num_hash_layers {
            Some(token_ids)
        } else {
            None
        };
        let want = stage_layers.contains(&layer_idx);
        if want {
            let mut trace = ParentLayerTrace::default();
            parent_layer_forward_traced(
                gpu,
                backend,
                weights,
                cfg,
                &mut layer_scratch,
                layer_idx,
                x,
                n,
                /*start_pos=*/ 0,
                input_ids,
                &rings[layer_idx],
                out,
                &mut trace,
            )?;
            let ratio = cfg.compress_ratio(layer_idx);
            println!(
                "PARENT_STAGE layer={layer_idx} ratio={ratio} \
                 hc_pre_attn={:.6} attn_norm={:.6} attn_out={:.6} hc_post_attn={:.6} \
                 hc_pre_ffn={:.6} ffn_norm={:.6} moe_out={:.6} hc_post_ffn={:.6}",
                trace.hc_pre_attn,
                trace.attn_norm,
                trace.attn_out,
                trace.hc_post_attn,
                trace.hc_pre_ffn,
                trace.ffn_norm,
                trace.moe_out,
                trace.hc_post_ffn
            );
            // Also residual L2 for the full HC state.
            let res = gpu
                .download_f32(out)
                .map_err(|e| format!("download out: {e:?}"))?;
            let need = n * PARENT_HC_DIM;
            let res_l2 = l2_f64(&res[..need.min(res.len())]);
            println!(
                "PARENT_STAGE layer={layer_idx} residual_out_l2={res_l2:.6} (nelems={need})"
            );
            // Indexer / compressed dump for ratio-4 layers.
            if ratio == 4 {
                dump_parent_indexer(gpu, &layer_scratch, layer_idx, n)?;
            }
        } else {
            hipfire_arch_deepseek4::parent::forward::parent_layer_forward(
                gpu,
                backend,
                weights,
                cfg,
                &mut layer_scratch,
                layer_idx,
                x,
                n,
                0,
                input_ids,
                &rings[layer_idx],
                out,
            )?;
        }
        use_a = !use_a;
    }
    Ok(())
}

fn dump_parent_indexer(
    gpu: &Gpu,
    layer_scratch: &ParentForwardScratch,
    layer_idx: usize,
    rows: usize,
) -> Result<(), String> {
    use hipfire_arch_deepseek4::parent::attention::{
        PARENT_ATTN_INDEX_TOPK, PARENT_HEAD_DIM,
    };
    use hipfire_arch_deepseek4::parent::indexer::PARENT_INDEX_HEAD_DIM;
    let attn = layer_scratch.attn_scratch();
    let n_comp = attn.last_compress_events();
    // topk_idx is I32 in Raw buffer.
    let topk_bytes = {
        let t = attn.topk_idx_ref();
        let nbytes = rows * PARENT_ATTN_INDEX_TOPK * 4;
        let mut buf = vec![0u8; nbytes];
        gpu.hip
            .memcpy_dtoh(&mut buf, &t.buf)
            .map_err(|e| format!("dtoh topk: {e:?}"))?;
        buf
    };
    let topk: Vec<i32> = topk_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // Print last row's first 32 indices (full context).
    let last = rows.saturating_sub(1);
    let base = last * PARENT_ATTN_INDEX_TOPK;
    let head: Vec<String> = topk[base..base + 32.min(PARENT_ATTN_INDEX_TOPK)]
        .iter()
        .map(|v| v.to_string())
        .collect();
    let n_pos = topk[base..base + PARENT_ATTN_INDEX_TOPK]
        .iter()
        .filter(|&&v| v >= 0)
        .count();
    // main_kv L2 over committed compressed slots.
    let main = gpu
        .download_f32(attn.main_kv_cache_ref())
        .map_err(|e| format!("main_kv: {e:?}"))?;
    let main_n = n_comp.saturating_mul(PARENT_HEAD_DIM).min(main.len());
    let main_l2 = l2_f64(&main[..main_n]);
    // indexer compressed cache
    let idx_kv = gpu
        .download_f32(attn.indexer_scratch().kv_cache_f32_ref())
        .map_err(|e| format!("idx_kv: {e:?}"))?;
    let idx_n = n_comp.saturating_mul(PARENT_INDEX_HEAD_DIM).min(idx_kv.len());
    let idx_l2 = l2_f64(&idx_kv[..idx_n]);
    println!(
        "PARENT_INDEXER layer={layer_idx} row={last} n_comp={n_comp} n_pos_topk={n_pos} \
         main_kv_l2={main_l2:.6e} idx_kv_l2={idx_l2:.6e} topk32=[{}]",
        head.join(",")
    );
    Ok(())
}


// ── compare (offline) ───────────────────────────────────────────────────────

fn run_compare(args: &Args) -> Result<(), String> {
    println!("=== ds4_prod_vs_parent_trace (compare / offline) ===");
    println!();
    println!("=== parent baseline 1024-token residual L2 (Gate-6 promoted) ===");
    print_trajectory("parent_1024", PARENT_BASELINE_1024);

    if let Some(prod_csv) = args.prod_csv.as_ref() {
        let prod = read_prod_series(prod_csv)?;
        println!();
        println!("=== production series from {} ===", prod_csv.display());
        print_trajectory("prod", &prod);
        println!();
        compare_ratio_trajectories(PARENT_BASELINE_1024, &prod);
    } else {
        println!();
        println!(
            "(no --prod-csv; run --mode prod first and pass the captured series, \
             or paste PROD_LAYER_NORM lines into a file)"
        );
    }

    print_structural_diff();
    print_embed_head_boundary();
    Ok(())
}

// ── trajectory helpers ──────────────────────────────────────────────────────

fn print_trajectory(tag: &str, norms: &[f64]) {
    if norms.is_empty() {
        println!("  ({tag}: empty)");
        return;
    }
    println!("  layer   L2({tag})           ratio_to_prev");
    for (i, &n) in norms.iter().enumerate() {
        if i == 0 {
            println!("  {i:>5}   {n:>16.6}   —");
        } else {
            let r = n / norms[i - 1];
            println!("  {i:>5}   {n:>16.6}   {r:.6}");
        }
    }
    let first = norms[0];
    let last = *norms.last().unwrap();
    let growth = last / first;
    let geo = if norms.len() > 1 {
        growth.powf(1.0 / (norms.len() as f64 - 1.0))
    } else {
        1.0
    };
    println!(
        "  summary: first={first:.6} last={last:.6} growth={growth:.4} \
         geo_mean_ratio={geo:.6} layers={}",
        norms.len()
    );
}

fn compare_ratio_trajectories(parent: &[f64], prod: &[f64]) {
    let n = parent.len().min(prod.len());
    if n < 2 {
        println!("compare: need ≥2 layers on both sides");
        return;
    }
    println!("=== consecutive-layer ratio compare (representation-independent) ===");
    println!("  layer   parent_r   prod_r    prod/parent");
    let mut first_sep: Option<usize> = None;
    for i in 1..n {
        let pr = parent[i] / parent[i - 1];
        let qr = prod[i] / prod[i - 1];
        let rel = if pr > 0.0 { qr / pr } else { f64::NAN };
        // Flag when ratio shapes diverge by >25% (quant is a few percent).
        let diverge = rel.is_finite() && !(0.75..=1.25).contains(&rel);
        if diverge && first_sep.is_none() {
            first_sep = Some(i);
        }
        let mark = if diverge { "  <--" } else { "" };
        println!("  {i:>5}   {pr:.6}   {qr:.6}   {rel:.4}{mark}");
    }
    match first_sep {
        Some(l) => println!(
            "first layer where consecutive-ratio shape diverges >25%: L{l}"
        ),
        None => println!(
            "ratio trajectories track within 25% across {} steps \
             (quantitative, not structural?)",
            n - 1
        ),
    }

    let pg = parent[n - 1] / parent[0];
    let qg = prod[n - 1] / prod[0];
    println!(
        "end-to-end growth: parent={pg:.4}  prod={qg:.4}  prod/parent={:.4}",
        qg / pg
    );
}

// ── structural diff (static, always printed) ────────────────────────────────

fn print_structural_diff() {
    println!();
    println!("=== structural layer diff (production vs parent) ===");
    println!(
        "One layer, step-by-step. Production is token-at-a-time decode; \
         parent is multi-row prefill. Ordering and presence of steps is what matters.\n"
    );
    println!("| # | production (`forward.rs`) | parent (`parent/forward.rs`) | note |");
    println!("|---|---------------------------|------------------------------|------|");
    println!(
        "| 1 | `mhc_pre(..., is_attn=true)` ~L2907/L3034 | `parent_hc_pre(hc_attn_*)` L483-498 | same role |"
    );
    println!(
        "| 2 | (inside mhc_pre) control + input map → `hc_x_in` | `parent_hc_pre` → `stream_y` | same |"
    );
    println!(
        "| 3 | `q_lora` (attn_norm fused inside) | `parent_rms_norm(attn_norm)` then attn | prod fuses norm |"
    );
    println!(
        "| 4 | `kv_joint` | (inside `parent_attention_swa`) | same |"
    );
    println!(
        "| 5 | `apply_tail_rope` | (inside attention) | same |"
    );
    println!(
        "| 6 | `compressor_forward` ratio>0 | (inside attention via ParentAttnScratch) | same |"
    );
    println!(
        "| 7 | `indexer_forward` ratio==4 | (inside attention) | same |"
    );
    println!(
        "| 8 | `attn_stub` | `parent_attention_swa` | same role |"
    );
    println!(
        "| 9 | `hc_attn_mix` | `parent_hc_post` | same: comb·res + post·attn |"
    );
    println!(
        "|10 | `mhc_pre(..., is_attn=false)` | `parent_hc_pre(hc_ffn_*)` | same |"
    );
    println!(
        "|11 | `ffn_stub` + hash/score routed | `parent_rms_norm(ffn)` + route + moe | same |"
    );
    println!(
        "|12 | `hc_ffn_mix` | `parent_hc_post` | same |"
    );
    println!();
    println!(
        "Steps present in BOTH: HC-pre attn, attn path, HC-post attn, HC-pre ffn, MoE, HC-post ffn."
    );
    println!("Constant / packing differences (not missing Block.forward steps):");
    println!(
        "  - production fuses attn_norm into `q_lora` / ffn_norm into `ffn_stub`; \
         parent calls `parent_rms_norm` explicitly."
    );
    println!(
        "  - production default `post_scale=1.5` in `mhc_pre` (forward.rs ~6231-6241); \
         parent/reference `2 * sigmoid` (hc.rs:148-150, kernel.py:394). \
         DO NOT import 1.5 into the parent reference."
    );
    println!(
        "  - production `route_scale` uses cfg.routed_scaling_factor unless \
         HIPFIRE_DEEPSEEK4_ROUTE_SCALE overrides; parent has no route_scale \
         (fp8 experts, scores already calibrated)."
    );
    println!(
        "  - production residual is single-token `[hc_mult, hidden]`; \
         parent is `[rows, hc_mult, hidden]`."
    );
    println!(
        "  - NO missing Block.forward step on either side relative to model.py:695-707."
    );
}

fn print_embed_head_boundary() {
    println!();
    println!("=== embed boundary ===");
    println!(
        "reference `Transformer.forward` model.py:914-916:\n\
         `h = embed(ids); h = h.unsqueeze(2).repeat(1,1,hc_mult,1)` — ALL streams get the embed."
    );
    println!(
        "production `init_residual_streams` forward.rs ~7122-7134:\n\
         copies embed into every stream 0..hc_mult (comment corrected from prior [embed,0,0,0])."
    );
    println!(
        "parent `parent_embed` head.rs:217-223:\n\
         BF16 gather → F32 widen → splat across hc_mult. MATCHES reference and production."
    );
    println!();
    println!("=== head boundary ===");
    println!(
        "reference model.py:922-923: `hc_head` (plain sigmoid, no sinkhorn) → `norm` → `ParallelHead`.\n\
         ParallelHead.forward model.py:731-735: if not full_logits: `x = x[:, -1]` before projection\n\
         — production decode always projects the current (last) token; parent prefill projects ALL rows."
    );
    println!(
        "production `final_norm_and_head_impl` / `final_norm_compute`:\n\
         hc_head_compute_pre → hc_input_map_4stream → rmsnorm → lm_head GEMV. Single position."
    );
    println!(
        "parent `parent_head` head.rs:291-301:\n\
         parent_hc_head → parent_rms_norm → BF16 GEMM head.weight. All `rows` positions.\n\
         For PPL/KLD the multi-row parent logits are the right contract; production plog\n\
         capture (ds4_quant_plog) loops decode_step to emit one row per position."
    );
    println!(
        "No structural missing head step. Row-count difference is intentional (prefill vs decode)."
    );
}

// ── I/O helpers ─────────────────────────────────────────────────────────────

fn zeros_f32(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.zeros(shape, DType::F32)
        .map_err(|e| format!("zeros {shape:?}: {e:?}"))
}

fn write_csv_series(path: &Path, norms: &[f64]) -> Result<(), String> {
    let mut f =
        std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    writeln!(f, "layer,l2").map_err(|e| e.to_string())?;
    for (i, &n) in norms.iter().enumerate() {
        writeln!(f, "{i},{n}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Read either a simple `layer,l2` CSV or a file of `PROD_LAYER_NORM ...` lines.
/// When PROD_LAYER_NORM lines contain multiple positions, keep the **last**
/// position's per-layer series (full context).
fn read_prod_series(path: &Path) -> Result<Vec<f64>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let reader = std::io::BufReader::new(f);
    let mut by_pos: std::collections::BTreeMap<u32, Vec<(usize, f64)>> =
        std::collections::BTreeMap::new();
    let mut simple: Vec<f64> = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("PROD_LAYER_NORM ") {
            let mut pos: Option<u32> = None;
            let mut layer: Option<usize> = None;
            let mut l2: Option<f64> = None;
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("pos=") {
                    pos = v.parse().ok();
                } else if let Some(v) = tok.strip_prefix("layer=") {
                    layer = v.parse().ok();
                } else if let Some(v) = tok.strip_prefix("l2=") {
                    l2 = v.parse().ok();
                }
            }
            if let (Some(p), Some(l), Some(v)) = (pos, layer, l2) {
                by_pos.entry(p).or_default().push((l, v));
            }
            continue;
        }
        if line.starts_with("layer") || line.starts_with("pos") {
            continue;
        }
        let parts: Vec<_> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Ok(v) = parts[1].parse::<f64>() {
                simple.push(v);
            }
        } else if let Ok(v) = line.parse::<f64>() {
            simple.push(v);
        }
    }
    if !by_pos.is_empty() {
        let (pos, mut rows) = by_pos.into_iter().next_back().unwrap();
        rows.sort_by_key(|(l, _)| *l);
        println!(
            "(using PROD_LAYER_NORM series at pos={pos}, {} layers)",
            rows.len()
        );
        return Ok(rows.into_iter().map(|(_, v)| v).collect());
    }
    if simple.is_empty() {
        return Err(format!("no layer norms parsed from {}", path.display()));
    }
    Ok(simple)
}

fn l2_f64(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| {
            let x = x as f64;
            x * x
        })
        .sum::<f64>()
        .sqrt()
}

fn read_token_ids(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "token-ids {} length {} not multiple of 4",
            path.display(),
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn eq_hex_ci(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .all(|(x, y)| x.to_ascii_lowercase() == y.to_ascii_lowercase())
}

// ── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mode: String,
    model: String,
    expect_sha256: Option<String>,
    token_ids: PathBuf,
    rows: usize,
    csv: Option<PathBuf>,
    prod_csv: Option<PathBuf>,
    stage_dump: bool,
    stage_layers: Vec<usize>,
    skip_sha256: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut mode = "compare".to_string();
    let mut model = String::new();
    let mut expect_sha256 = None;
    let mut token_ids = PathBuf::from(DEFAULT_TOKEN_IDS);
    let mut rows = DEFAULT_ROWS;
    let mut csv = None;
    let mut prod_csv = None;
    let mut skip_sha256 = false;
    let mut stage_dump = false;
    let mut stage_layers: Vec<usize> = vec![25, 26, 27];

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--mode" => {
                mode = args.next().ok_or("--mode needs value")?;
            }
            "--model" => {
                model = args.next().ok_or("--model needs value")?;
            }
            "--expect-sha256" => {
                expect_sha256 = Some(args.next().ok_or("--expect-sha256 needs value")?);
            }
            "--token-ids" => {
                token_ids = PathBuf::from(args.next().ok_or("--token-ids needs value")?);
            }
            "--rows" => {
                rows = args
                    .next()
                    .ok_or("--rows needs value")?
                    .parse()
                    .map_err(|e| format!("--rows: {e}"))?;
            }
            "--csv" => {
                csv = Some(PathBuf::from(args.next().ok_or("--csv needs value")?));
            }
            "--prod-csv" => {
                prod_csv = Some(PathBuf::from(args.next().ok_or("--prod-csv needs value")?));
            }
            "--skip-sha256" => {
                skip_sha256 = true;
            }
            "--stage-dump" => {
                stage_dump = true;
            }
            "--stage-layers" => {
                let s = args.next().ok_or("--stage-layers needs value")?;
                stage_layers = s
                    .split(',')
                    .map(|t| t.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("--stage-layers: {e}"))?;
                stage_dump = true;
            }
            "-h" | "--help" => {
                eprintln!(
                    "ds4_prod_vs_parent_trace --mode prod|parent|compare [options]\n\
                     --model PATH  --expect-sha256 HEX  --token-ids PATH  --rows N\n\
                     --csv PATH  --prod-csv PATH  --skip-sha256\n\
                     --stage-dump  --stage-layers 25,26,27"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }

    if model.is_empty() {
        model = match mode.as_str() {
            "prod" => DEFAULT_MQ2R.to_string(),
            "parent" => DEFAULT_PARENT_MODEL.to_string(),
            _ => String::new(),
        };
    }

    Ok(Args {
        mode,
        model,
        expect_sha256,
        token_ids,
        rows,
        csv,
        prod_csv,
        skip_sha256,
        stage_dump,
        stage_layers,
    })
}
