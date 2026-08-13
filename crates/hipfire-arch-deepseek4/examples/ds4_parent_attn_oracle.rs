// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! GPU parent attention vs f64 joint SWA+compressed `attention_swa_ref` oracle.
//!
//! Compares the full attention path at 128 tokens for:
//! - layer 0  (`compress_ratio == 0`, plain RoPE base 10000) — **floor check**
//! - layer 2  (`compress_ratio == 4`, YaRN + 32 compress events + indexer)
//! - layer 3  (`compress_ratio == 128`, YaRN + 1 compress event, identity)
//!
//! The oracle models the joint softmax over SWA window + compressed keys +
//! sink (`model.py:520,531-533`). Layer 0 must reproduce the known ~1e-6
//! floor before layers 2/3 are interpreted.
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_attn_oracle \
//!   -- --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 --rows 128
//! ```
use hipfire_arch_deepseek4::parent::attention::{
    all_finite, l2_norm, parent_attention_swa, precompute_rope_freqs, ParentAttnScratch, PARENT_DIM,
    PARENT_HEAD_DIM, PARENT_N_HEADS, PARENT_N_KV_HEADS, PARENT_O_GROUPS, PARENT_O_LORA,
    PARENT_PER_GROUP_IN, PARENT_Q_LORA, PARENT_Q_WIDTH, PARENT_ROPE_DIM, PARENT_ROPE_THETA,
    PARENT_SWA_WINDOW, PARENT_WO_A_OUT,
};
use hipfire_arch_deepseek4::parent::codec::{
    act_quant_fp8_inplace_ref, fast_round_scale, round_to_bf16,
};
use hipfire_arch_deepseek4::parent::compressor::{
    PARENT_COMPRESS_ROPE_THETA, PARENT_YARN_BETA_FAST, PARENT_YARN_BETA_SLOW, PARENT_YARN_FACTOR,
    PARENT_YARN_ORIG_SEQ,
};
use hipfire_arch_deepseek4::parent::inventory::ParentInventory;
use hipfire_arch_deepseek4::parent::layer_ref::{
    attention_main_rope_policy, attention_swa_ref, AttnCompRefWeights, AttnIndexerRefWeights,
    AttnSwARefWeights,
};
use hipfire_arch_deepseek4::parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_arch_deepseek4::parent::{Ds4ParentBackend, ParentQuantConfig};
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const DEFAULT_ROWS: usize = 128;
const START_POS: usize = 0;
const REPORT_ROWS: &[usize] = &[0, 1, 8, 32, 64, 100, 127];
const LAYERS: &[(usize, usize)] = &[(0, 0), (2, 4), (3, 128)];
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
    let (model, rows) = parse_args();
    let model_path = Path::new(&model);
    if !model_path.is_dir() {
        return Err(format!(
            "deepseek4 parent: --model must be a directory, got {}",
            model_path.display()
        ));
    }
    println!("=== ds4_parent_attn_oracle ===");
    println!("model: {}", model_path.display());
    println!("rows: {rows}  start_pos: {START_POS}");
    println!("layers: {:?}", LAYERS);

    // ── RoPE table verification (code + numeric) ────────────────────────
    verify_rope_tables()?;

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
    let (backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    for &(layer_idx, expect_ratio) in LAYERS {
        let got = cfg.compress_ratio(layer_idx);
        if got != expect_ratio {
            return Err(format!(
                "deepseek4 parent: expected layer {layer_idx} compress_ratio={expect_ratio}, got {got}"
            ));
        }
    }
    let inv = ParentInventory::build(&source, &cfg)?;

    // Deterministic post-attn_norm activations (same grid as attn_smoke).
    let mut x_f32 = vec![0.0f32; rows * PARENT_DIM];
    for r in 0..rows {
        for k in 0..PARENT_DIM {
            let v = (((r * 131 + k * 17) % 200) as f32 - 100.0) * 0.01;
            x_f32[r * PARENT_DIM + k] = round_to_bf16(v);
        }
    }

    let mut any_implicated = false;
    for &(layer_idx, ratio) in LAYERS {
        println!();
        println!(
            "################################################################"
        );
        println!("# layer {layer_idx}  compress_ratio={ratio}");
        println!(
            "################################################################"
        );
        let implicated = run_layer(
            &mut gpu,
            backend,
            &source,
            &cfg,
            &inv,
            &x_f32,
            rows,
            layer_idx,
            ratio,
        )?;
        any_implicated |= implicated;
    }

    println!();
    if any_implicated {
        println!(
            "OVERALL: at least one ratio>0 layer shows position-growing main-path error."
        );
    } else {
        println!(
            "OVERALL: main-path stages agree with oracle on all reported layers \
             (ratio>0 RoPE table is consumed correctly on q/kv/inv)."
        );
    }
    Ok(())
}

/// Confirm both frequency tables and document which policy the GPU main path uses.
fn verify_rope_tables() -> Result<(), String> {
    let (o0, t0) = attention_main_rope_policy(0)?;
    let (o4, t4) = attention_main_rope_policy(4)?;
    let (o128, t128) = attention_main_rope_policy(128)?;
    assert_eq!((o4, t4), (o128, t128));
    assert_eq!(o0, 0);
    assert!((t0 - PARENT_ROPE_THETA as f64).abs() < 1e-12);
    assert_eq!(o4, PARENT_YARN_ORIG_SEQ);
    assert!((t4 - PARENT_COMPRESS_ROPE_THETA).abs() < 1e-12);

    let plain = precompute_rope_freqs(
        PARENT_ROPE_DIM,
        o0,
        t0,
        PARENT_YARN_FACTOR,
        PARENT_YARN_BETA_FAST,
        PARENT_YARN_BETA_SLOW,
    )?;
    let yarn = precompute_rope_freqs(
        PARENT_ROPE_DIM,
        o4,
        t4,
        PARENT_YARN_FACTOR,
        PARENT_YARN_BETA_FAST,
        PARENT_YARN_BETA_SLOW,
    )?;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for (a, b) in plain.iter().zip(yarn.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        let r = d / a.abs().max(1e-30);
        if r > max_rel {
            max_rel = r;
        }
    }
    println!();
    println!("=== RoPE table policy (model.py:481-488 / attention.rs:1000-1021) ===");
    println!("ratio==0: original_seq_len={o0}  theta={t0}  (YaRN off)");
    println!("ratio>0:  original_seq_len={o4}  theta={t4}  (YaRN on)");
    println!(
        "table divergence plain vs yarn: max_abs={max_abs:.6e}  max_rel={max_rel:.6e}"
    );
    println!(
        "GPU parent_attention_swa call sites for ratio>0 (all three share one `freqs`):"
    );
    println!("  - main q   apply_rope_interleaved_inplace(..., inverse=false)  // ~L1023");
    println!("  - main kv  apply_rope_interleaved_inplace(..., inverse=false)  // ~L1033");
    println!("  - inv o    apply_rope_interleaved_inplace(..., inverse=true)   // ~L1157");
    println!(
        "  selection: if ratio==0 {{ plain 10000 }} else {{ YaRN + compress_rope_theta 160000 }}"
    );
    if max_rel < 0.1 {
        return Err(format!(
            "deepseek4 parent: expected plain vs yarn tables to differ (max_rel={max_rel})"
        ));
    }
    // Spot-check first freq: plain 1/10000^(0)=1; yarn blends with /factor.
    println!(
        "freqs[0]: plain={:.8e}  yarn={:.8e}  (pos=100 angle delta={:.6e})",
        plain[0],
        yarn[0],
        100.0 * (plain[0] - yarn[0]).abs()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_layer(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    source: &SafetensorsSource,
    cfg: &ParentQuantConfig,
    inv: &ParentInventory,
    x_f32: &[f32],
    rows: usize,
    layer_idx: usize,
    ratio: usize,
) -> Result<bool, String> {
    // Free prior layer before loading the next (parent is huge).
    let plan = ParentLoadPlan {
        layers: layer_idx..(layer_idx + 1),
        load_experts: false,
    };
    let t_load = Instant::now();
    let weights = ParentWeights::load(source, cfg, inv, gpu, backend, &plan)?;
    println!(
        "loaded layer {layer_idx} in {:.2}s  resident={:.3} GiB",
        t_load.elapsed().as_secs_f64(),
        weights.residency().total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let layer = weights
        .layers
        .iter()
        .find(|l| l.layer_idx == layer_idx)
        .ok_or_else(|| format!("deepseek4 parent: layer {layer_idx} missing after load"))?;
    if layer.compress_ratio != ratio {
        return Err(format!(
            "deepseek4 parent: layer {layer_idx} weight compress_ratio={} != expected {ratio}",
            layer.compress_ratio
        ));
    }

    let wq_a = download_bf16_as_f32(gpu, layer.wq_a.tensor(), layer.wq_a.n() * layer.wq_a.k())?;
    let wq_b = download_bf16_as_f32(gpu, layer.wq_b.tensor(), layer.wq_b.n() * layer.wq_b.k())?;
    let wkv = download_bf16_as_f32(gpu, layer.wkv.tensor(), layer.wkv.n() * layer.wkv.k())?;
    let wo_a = download_bf16_as_f32(gpu, layer.wo_a.tensor(), layer.wo_a.n() * layer.wo_a.k())?;
    let wo_b = download_bf16_as_f32(gpu, layer.wo_b.tensor(), layer.wo_b.n() * layer.wo_b.k())?;
    let q_norm = download_bf16_as_f32(gpu, &layer.q_norm, PARENT_Q_LORA)?;
    let kv_norm = download_bf16_as_f32(gpu, &layer.kv_norm, PARENT_HEAD_DIM)?;
    let attn_sink = download_f32(gpu, &layer.attn_sink, PARENT_N_HEADS)?;
    println!(
        "weights: wq_a=[{},{}] wq_b=[{},{}] wkv=[{},{}] wo_a=[{},{}] wo_b=[{},{}]",
        layer.wq_a.n(),
        layer.wq_a.k(),
        layer.wq_b.n(),
        layer.wq_b.k(),
        layer.wkv.n(),
        layer.wkv.k(),
        layer.wo_a.n(),
        layer.wo_a.k(),
        layer.wo_b.n(),
        layer.wo_b.k()
    );
    assert_eq!(layer.wq_a.n(), PARENT_Q_LORA);
    assert_eq!(layer.wq_a.k(), PARENT_DIM);
    assert_eq!(layer.wq_b.n(), PARENT_Q_WIDTH);
    assert_eq!(layer.wq_b.k(), PARENT_Q_LORA);
    assert_eq!(layer.wkv.n(), PARENT_HEAD_DIM);
    assert_eq!(layer.wkv.k(), PARENT_DIM);
    assert_eq!(layer.wo_a.n(), PARENT_WO_A_OUT);
    assert_eq!(layer.wo_a.k(), PARENT_PER_GROUP_IN);
    assert_eq!(layer.wo_b.n(), PARENT_DIM);
    assert_eq!(layer.wo_b.k(), PARENT_WO_A_OUT);
    let _ = (PARENT_O_GROUPS, PARENT_O_LORA, PARENT_N_KV_HEADS);

    // Optional compressor / indexer weights for the joint oracle.
    let (comp_wkv, comp_wgate, comp_norm, comp_ape) = if let Some(c) = layer.compressor.as_ref() {
        let proj = c.wkv.shape.get(0).copied().unwrap_or(0);
        let dim_k = c.wkv.shape.get(1).copied().unwrap_or(PARENT_DIM);
        let wkv_c = download_bf16_as_f32(gpu, &c.wkv, proj * dim_k)?;
        let wgate_c = download_bf16_as_f32(gpu, &c.wgate, proj * dim_k)?;
        let norm_c = download_bf16_as_f32(gpu, &c.norm, PARENT_HEAD_DIM)?;
        let ape_n = c.ape.shape.iter().product::<usize>().max(1);
        let ape_c = download_f32(gpu, &c.ape, ape_n)?;
        println!(
            "compressor: wkv=[{proj},{dim_k}] ape_elems={ape_n} ratio={ratio}"
        );
        (Some(wkv_c), Some(wgate_c), Some(norm_c), Some(ape_c))
    } else {
        (None, None, None, None)
    };
    let (ix_wq_b, ix_wp, ix_wkv, ix_wgate, ix_norm, ix_ape) =
        if let Some(ix) = layer.indexer.as_ref() {
            let wq = download_bf16_as_f32(gpu, ix.wq_b.tensor(), ix.wq_b.n() * ix.wq_b.k())?;
            let wp_n = ix.weights_proj.shape.get(0).copied().unwrap_or(0);
            let wp_k = ix.weights_proj.shape.get(1).copied().unwrap_or(PARENT_DIM);
            let wp = download_bf16_as_f32(gpu, &ix.weights_proj, wp_n * wp_k)?;
            let cproj = ix.compressor_wkv.shape.get(0).copied().unwrap_or(0);
            let cdim = ix.compressor_wkv.shape.get(1).copied().unwrap_or(PARENT_DIM);
            let cwkv = download_bf16_as_f32(gpu, &ix.compressor_wkv, cproj * cdim)?;
            let cwgate = download_bf16_as_f32(gpu, &ix.compressor_wgate, cproj * cdim)?;
            // index head_dim = 128
            let cnorm = download_bf16_as_f32(gpu, &ix.compressor_norm, 128)?;
            let cape_n = ix.compressor_ape.shape.iter().product::<usize>().max(1);
            let cape = download_f32(gpu, &ix.compressor_ape, cape_n)?;
            println!(
                "indexer: wq_b=[{},{}] weights_proj=[{wp_n},{wp_k}] comp_wkv=[{cproj},{cdim}]",
                ix.wq_b.n(),
                ix.wq_b.k()
            );
            (
                Some(wq),
                Some(wp),
                Some(cwkv),
                Some(cwgate),
                Some(cnorm),
                Some(cape),
            )
        } else {
            (None, None, None, None, None, None)
        };

    let (orig, theta) = attention_main_rope_policy(ratio)?;
    println!(
        "oracle RoPE policy: original_seq_len={orig}  theta={theta}  (ratio={ratio})"
    );

    let comp_ref = match (
        comp_wkv.as_deref(),
        comp_wgate.as_deref(),
        comp_norm.as_deref(),
        comp_ape.as_deref(),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => Some(AttnCompRefWeights {
            wkv: a,
            wgate: b,
            norm: c,
            ape: d,
        }),
        _ => None,
    };
    let ix_ref = match (
        ix_wq_b.as_deref(),
        ix_wp.as_deref(),
        ix_wkv.as_deref(),
        ix_wgate.as_deref(),
        ix_norm.as_deref(),
        ix_ape.as_deref(),
    ) {
        (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) => Some(AttnIndexerRefWeights {
            wq_b: a,
            weights_proj: b,
            compressor_wkv: c,
            compressor_wgate: d,
            compressor_norm: e,
            compressor_ape: f,
        }),
        _ => None,
    };

    let wref = AttnSwARefWeights {
        wq_a: &wq_a,
        wq_b: &wq_b,
        wkv: &wkv,
        wo_a: &wo_a,
        wo_b: &wo_b,
        q_norm: &q_norm,
        kv_norm: &kv_norm,
        attn_sink: &attn_sink,
        compressor: comp_ref,
        indexer: ix_ref,
    };
    let t_ref = Instant::now();
    let reference = attention_swa_ref(x_f32, &wref, rows, START_POS, ratio)?;
    println!(
        "oracle done in {:.2}s  o_finite={} o_l2={:.6}  n_comp={} k_comp_stride={}",
        t_ref.elapsed().as_secs_f64(),
        all_finite(&reference.o),
        l2_norm(&reference.o),
        if PARENT_HEAD_DIM > 0 {
            reference.kv_compress.len() / PARENT_HEAD_DIM
        } else {
            0
        },
        if rows > 0 {
            reference.compress_idxs.len() / rows
        } else {
            0
        }
    );
    // ── GPU path ────────────────────────────────────────────────────────
    let mut scratch = ParentAttnScratch::new(gpu, cfg, rows)?;
    let kv_ring = gpu
        .zeros(
            &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
            DType::F32,
        )
        .map_err(|e| format!("deepseek4 parent: kv_ring alloc: {e:?}"))?;
    let x = upload_f32(gpu, x_f32, &[rows, PARENT_DIM])?;
    let out = gpu
        .zeros(&[rows, PARENT_DIM], DType::F32)
        .map_err(|e| format!("deepseek4 parent: out alloc: {e:?}"))?;

    let t_gpu = Instant::now();
    parent_attention_swa(
        gpu,
        backend,
        layer,
        cfg,
        &mut scratch,
        &x,
        rows,
        START_POS,
        &kv_ring,
        &out,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("deepseek4 parent: sync: {e:?}"))?;
    println!(
        "gpu forward in {:.2}s  last_compress_events={}",
        t_gpu.elapsed().as_secs_f64(),
        scratch.last_compress_events()
    );

    let gpu_o = download_f32(gpu, &out, rows * PARENT_DIM)?;
    let gpu_q = download_f32(gpu, scratch.q_f32_ref()?, rows * PARENT_Q_WIDTH)?;
    let gpu_kv = download_f32(gpu, scratch.kv_f32_ref()?, rows * PARENT_HEAD_DIM)?;
    // After forward, attn_out_f32 holds post-inverse-RoPE (before wo_a).
    let gpu_attn_inv = download_f32(gpu, scratch.attn_out_f32_ref()?, rows * PARENT_Q_WIDTH)?;
    let gpu_wo_a = download_f32(gpu, scratch.wo_a_out_f32_ref()?, rows * PARENT_WO_A_OUT)?;

    if !all_finite(&gpu_o) {
        return Err(format!(
            "deepseek4 parent: GPU output non-finite on layer {layer_idx}"
        ));
    }

    // ── Per-row divergence tables ───────────────────────────────────────
    println!();
    println!("=== final output o[rows, dim] GPU vs JOINT oracle (SWA+compress+sink) ===");
    if ratio == 0 {
        println!("(floor check: ratio==0 must stay ~1e-6 flat across rows)");
    } else {
        println!(
            "(joint path: n_comp={}, compress events this call={})",
            if PARENT_HEAD_DIM > 0 {
                reference.kv_compress.len() / PARENT_HEAD_DIM
            } else {
                0
            },
            scratch.last_compress_events()
        );
    }
    print_row_table("o", &gpu_o, &reference.o, rows, PARENT_DIM, REPORT_ROWS);

    println!();
    println!("=== stage: q_post_rope (GPU q_f32 after forward = post-RoPE q) ===");
    print_row_table(
        "q_rope",
        &gpu_q,
        &reference.q_post_rope,
        rows,
        PARENT_Q_WIDTH,
        REPORT_ROWS,
    );

    println!();
    println!("=== stage: kv_post_quant (GPU kv_f32 after forward) ===");
    print_row_table(
        "kv_q",
        &gpu_kv,
        &reference.kv_post_quant,
        rows,
        PARENT_HEAD_DIM,
        REPORT_ROWS,
    );

    println!();
    println!("=== stage: attn_inv_rope (GPU attn_out after forward = post-inv-RoPE) ===");
    print_row_table(
        "attn_inv",
        &gpu_attn_inv,
        &reference.attn_inv_rope,
        rows,
        PARENT_Q_WIDTH,
        REPORT_ROWS,
    );

    println!();
    println!("=== stage: wo_a_out (after grouped wo_a, before wo_b) ===");
    print_row_table(
        "wo_a",
        &gpu_wo_a,
        &reference.wo_a_out,
        rows,
        PARENT_WO_A_OUT,
        REPORT_ROWS,
    );

    // Full 128-row scan — sample tables already misled once.
    println!();
    println!("=== FULL per-row max_abs scan (all {rows} rows) ===");

    // Act-quant scale diagnostic on wo_b input (wo_a_out) for dirty vs clean rows.
    // Compare host oracle amax/scale against what act_quant_fp8_inplace_ref produces.
    if ratio == 0 {
        dump_act_quant_scales(
            "wo_b_input",
            &gpu_wo_a,
            &reference.wo_a_out,
            rows,
            PARENT_WO_A_OUT,
            128,
            &[0usize, 1, 7, 8, 16, 46, 68, 100, 127],
        );
        dump_act_quant_scales(
            "wo_a_input_attn_inv",
            &gpu_attn_inv,
            &reference.attn_inv_rope,
            rows,
            PARENT_Q_WIDTH,
            128,
            &[0usize, 1, 7, 8, 16, 46, 68, 100, 127],
        );
    }
    full_row_scan("o", &gpu_o, &reference.o, rows, PARENT_DIM);
    full_row_scan("attn_inv", &gpu_attn_inv, &reference.attn_inv_rope, rows, PARENT_Q_WIDTH);
    full_row_scan("wo_a", &gpu_wo_a, &reference.wo_a_out, rows, PARENT_WO_A_OUT);

    // Global summary
    let (gmax, gmean, gl2) = metrics(&gpu_o, &reference.o);
    println!();
    println!("GLOBAL final: max_abs={gmax:.6e}  mean_rel={gmean:.6e}  l2_rel={gl2:.6e}");
    let (qmax, _, ql2) = metrics(&gpu_q, &reference.q_post_rope);
    let (kmax, _, kl2) = metrics(&gpu_kv, &reference.kv_post_quant);
    let (amax, _, al2) = metrics(&gpu_attn_inv, &reference.attn_inv_rope);
    let (wamax, _, wal2) = metrics(&gpu_wo_a, &reference.wo_a_out);
    println!("GLOBAL q_post_rope:   max_abs={qmax:.6e}  l2_rel={ql2:.6e}");
    println!("GLOBAL kv_post_quant: max_abs={kmax:.6e}  l2_rel={kl2:.6e}");
    println!("GLOBAL attn_inv_rope: max_abs={amax:.6e}  l2_rel={al2:.6e}");
    println!("GLOBAL wo_a_out:      max_abs={wamax:.6e}  l2_rel={wal2:.6e}");

    // Flat-vs-growing diagnostic on MAIN-PATH stages (q and kv), not final o.
    let q_low = row_max_abs(&gpu_q, &reference.q_post_rope, 0, PARENT_Q_WIDTH);
    let high_r = if rows > 100 { 100 } else { rows - 1 };
    let q_high = row_max_abs(&gpu_q, &reference.q_post_rope, high_r, PARENT_Q_WIDTH);
    let kv_low = row_max_abs(&gpu_kv, &reference.kv_post_quant, 0, PARENT_HEAD_DIM);
    let kv_high = row_max_abs(&gpu_kv, &reference.kv_post_quant, high_r, PARENT_HEAD_DIM);
    let inv_low = row_max_abs(&gpu_attn_inv, &reference.attn_inv_rope, 0, PARENT_Q_WIDTH);
    let inv_high = row_max_abs(
        &gpu_attn_inv,
        &reference.attn_inv_rope,
        high_r,
        PARENT_Q_WIDTH,
    );
    println!();
    println!(
        "position signature q_rope:  row0={q_low:.6e}  row{high_r}={q_high:.6e}  ratio={:.3}",
        if q_low > 0.0 {
            q_high / q_low
        } else {
            f64::INFINITY
        }
    );
    println!(
        "position signature kv_q:    row0={kv_low:.6e}  row{high_r}={kv_high:.6e}  ratio={:.3}",
        if kv_low > 0.0 {
            kv_high / kv_low
        } else {
            f64::INFINITY
        }
    );
    println!(
        "position signature attn_inv: row0={inv_low:.6e}  row{high_r}={inv_high:.6e}  ratio={:.3}",
        if inv_low > 0.0 {
            inv_high / inv_low
        } else {
            f64::INFINITY
        }
    );

    // Joint-path cleanliness: for ratio==0, o/attn_inv must match the ~1e-6 floor.
    // For ratio>0, judge o and attn_inv (the previously un-oracled joint path).
    let o_low = row_max_abs(&gpu_o, &reference.o, 0, PARENT_DIM);
    let o_high = row_max_abs(&gpu_o, &reference.o, high_r, PARENT_DIM);
    println!(
        "position signature final_o:  row0={o_low:.6e}  row{high_r}={o_high:.6e}  ratio={:.3}",
        if o_low > 0.0 {
            o_high / o_low
        } else {
            f64::INFINITY
        }
    );

    let main_high = q_high.max(kv_high);
    let _main_low = q_low.max(kv_low);
    let joint_high = inv_high.max(o_high);
    let joint_low = inv_low.max(o_low);
    let implicated = joint_high > 1e-2 && joint_high > joint_low * 5.0;
    if ratio == 0 {
        if joint_high < 2e-5 && main_high < 2e-5 {
            println!(
                "VERDICT layer {layer_idx} FLOOR: joint SWA oracle agrees ~1e-6 flat (row0={o_low:.3e} row{high_r}={o_high:.3e})."
            );
        } else {
            println!(
                "VERDICT layer {layer_idx} FLOOR FAIL: expected ~1e-6, got o row0={o_low:.3e} row{high_r}={o_high:.3e}."
            );
        }
    } else if joint_high < 1e-3 && main_high < 1e-3 {
        println!(
            "VERDICT layer {layer_idx}: joint SWA+compress path agrees with oracle (clean)."
        );
    } else if implicated {
        println!(
            "VERDICT layer {layer_idx}: joint-path error GROWS with position \
             (row{high_r}/row0≈{:.1}x on o/attn_inv) — compress path implicated.",
            joint_high / joint_low.max(1e-30)
        );
    } else if joint_high > 1e-2 {
        println!(
            "VERDICT layer {layer_idx}: joint-path error is FLAT/elevated \
             (ratio={:.2}) — inspect compress keys/idxs.",
            joint_high / joint_low.max(1e-30)
        );
    } else {
        println!(
            "VERDICT layer {layer_idx}: mixed / small residual — inspect the per-row table."
        );
    }

    // Explicit numeric confirmation that GPU used the expected table:
    // if we *wrongly* compared GPU(ratio>0) against a plain-table oracle, q would diverge.
    if ratio > 0 {
        let wrong = attention_swa_ref(x_f32, &wref, rows, START_POS, /*compress_ratio=*/ 0)?;
        let (wmax, _, _) = metrics(&gpu_q, &wrong.q_post_rope);
        let (rmax, _, _) = metrics(&gpu_q, &reference.q_post_rope);
        println!(
            "RoPE end-to-end check: GPU q vs CORRECT yarn-oracle max_abs={rmax:.6e}; \
             GPU q vs WRONG plain-oracle max_abs={wmax:.6e}"
        );
        if rmax > 1e-2 {
            println!(
                "  → GPU main q does NOT match the yarn table oracle (bug or earlier stage)."
            );
        } else if wmax < rmax * 2.0 {
            println!(
                "  → plain vs yarn oracles too close on this input (unexpected; tables should separate)."
            );
        } else {
            println!(
                "  → GPU main q matches yarn table and rejects plain table (swap ruled out)."
            );
        }
    }

    // Drop weights/scratch by going out of scope.
    let _ = (weights, scratch, kv_ring, x, out);
    Ok(implicated)
}

fn parse_args() -> (String, usize) {
    let mut model = DEFAULT_MODEL.to_owned();
    let mut rows = DEFAULT_ROWS;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if let Some(p) = args.get(i + 1) {
                    model = p.clone();
                    i += 2;
                    continue;
                }
            }
            "--rows" => {
                if let Some(p) = args.get(i + 1) {
                    rows = p.parse().unwrap_or(DEFAULT_ROWS);
                    i += 2;
                    continue;
                }
            }
            s if !s.starts_with('-') => {
                model = s.to_owned();
            }
            _ => {}
        }
        i += 1;
    }
    let _ = REPORT_ROWS;
    (model, rows)
}

fn print_row_table(
    name: &str,
    gpu: &[f32],
    reference: &[f32],
    rows: usize,
    width: usize,
    report: &[usize],
) {
    println!(
        "{name:>10}  {:>6}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "row", "max_abs", "mean_rel", "l2_rel", "gpu_l2", "ref_l2"
    );
    for &r in report {
        if r >= rows {
            continue;
        }
        let g = &gpu[r * width..(r + 1) * width];
        let rf = &reference[r * width..(r + 1) * width];
        let (mx, mean_r, l2r) = metrics(g, rf);
        println!(
            "{name:>10}  {r:>6}  {mx:>12.6e}  {mean_r:>12.6e}  {l2r:>12.6e}  {:>12.6e}  {:>12.6e}",
            l2_norm(g) as f64,
            l2_norm(rf) as f64
        );
    }
}

/// Compare per-block amax + UE8M0 scale of GPU vs ref activations before a dense linear.
fn dump_act_quant_scales(
    name: &str,
    gpu: &[f32],
    reference: &[f32],
    rows: usize,
    width: usize,
    block: usize,
    report: &[usize],
) {
    println!();
    println!("=== act_quant scale probe: {name} (block={block}) ===");
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>12} {:>8} {:>6} {:>12} {:>12}",
        "row",
        "in_maxabs",
        "bf16_neq",
        "n_diff_sc",
        "max_amax_rel",
        "sc0",
        "expΔ",
        "bf16_maxabs",
        "q_out_maxabs"
    );
    for &r in report {
        if r >= rows {
            continue;
        }
        let g = &gpu[r * width..(r + 1) * width];
        let rf = &reference[r * width..(r + 1) * width];
        let in_mx = metrics(g, rf).0;

        let g_bf: Vec<f32> = g.iter().map(|&v| round_to_bf16(v)).collect();
        let r_bf: Vec<f32> = rf.iter().map(|&v| round_to_bf16(v)).collect();
        let mut bf16_neq = 0usize;
        let mut bf16_mx = 0.0f64;
        let mut first_mismatch: Option<(usize, f32, f32, f32, f32)> = None;
        for i in 0..width {
            if g_bf[i].to_bits() != r_bf[i].to_bits() {
                bf16_neq += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, g[i], rf[i], g_bf[i], r_bf[i]));
                }
            }
            let d = (g_bf[i] as f64 - r_bf[i] as f64).abs();
            if d > bf16_mx {
                bf16_mx = d;
            }
        }
        if let Some((i, gf, rf, gb, rb)) = first_mismatch {
            println!(
                "    first_bf16_mismatch idx={i} f32=({gf:.8e},{rf:.8e}) d_f32={:.3e} bf16=({gb:.8e},{rb:.8e}) d_bf16={:.3e}",
                (gf as f64 - rf as f64).abs(),
                (gb as f64 - rb as f64).abs()
            );
        }

        let n_blocks = width / block;
        let mut n_diff = 0usize;
        let mut max_amax_rel = 0.0f64;
        let mut max_exp_delta = 0i32;
        let mut sc0 = 0.0f32;
        for b in 0..n_blocks {
            let gs = &g_bf[b * block..(b + 1) * block];
            let rs = &r_bf[b * block..(b + 1) * block];
            let mut ag = 0.0f32;
            let mut ar = 0.0f32;
            for &v in gs {
                ag = ag.max(v.abs());
            }
            for &v in rs {
                ar = ar.max(v.abs());
            }
            ag = ag.max(1e-4);
            ar = ar.max(1e-4);
            let sg = fast_round_scale(ag, 1.0 / 448.0);
            let sr = fast_round_scale(ar, 1.0 / 448.0);
            if b == 0 {
                sc0 = sg;
            }
            if sg != sr {
                n_diff += 1;
                let eg = sg.log2().round() as i32;
                let er = sr.log2().round() as i32;
                max_exp_delta = max_exp_delta.max((eg - er).abs());
            }
            let rel = ((ag as f64) - (ar as f64)).abs() / (ar as f64).max(1e-30);
            if rel > max_amax_rel {
                max_amax_rel = rel;
            }
        }

        let mut g_q = g_bf.clone();
        let mut r_q = r_bf.clone();
        let _ = act_quant_fp8_inplace_ref(&mut g_q, width, block);
        let _ = act_quant_fp8_inplace_ref(&mut r_q, width, block);
        let q_mx = metrics(&g_q, &r_q).0;

        println!(
            "{r:>6} {in_mx:>10.3e} {bf16_neq:>10} {n_diff:>10} {max_amax_rel:>12.3e} {sc0:>8.3e} {max_exp_delta:>6} {bf16_mx:>12.3e} {q_mx:>12.3e}"
        );
    }
}

/// Scan every row; print dirty ones (max_abs > floor) and a shape summary.
fn full_row_scan(name: &str, gpu: &[f32], reference: &[f32], rows: usize, width: usize) {
    const FLOOR: f64 = 5e-6;
    let mut dirty: Vec<(usize, f64)> = Vec::new();
    let mut clean = 0usize;
    let mut max_all = 0.0f64;
    let mut argmax = 0usize;
    let mut per_row = vec![0.0f64; rows];
    for r in 0..rows {
        let mx = row_max_abs(gpu, reference, r, width);
        per_row[r] = mx;
        if mx > max_all {
            max_all = mx;
            argmax = r;
        }
        if mx > FLOOR {
            dirty.push((r, mx));
        } else {
            clean += 1;
        }
    }
    println!(
        "{name}: clean={clean}/{rows} (floor={FLOOR:.0e})  dirty={}  global_max={max_all:.6e} @row{argmax}",
        dirty.len()
    );
    if dirty.is_empty() {
        return;
    }
    // Print every dirty row (capped) so the pattern is visible.
    let show = dirty.len().min(64);
    print!("{name} dirty rows:");
    for i in 0..show {
        let (r, mx) = dirty[i];
        print!(" {r}:{mx:.3e}");
    }
    if dirty.len() > show {
        print!(" ...+{}", dirty.len() - show);
    }
    println!();
    // Periodicity probes: count dirty per residue mod 8 / 16 / 32 / 64.
    for &m in &[8usize, 16, 32, 64] {
        let mut bins = vec![0usize; m];
        for &(r, _) in &dirty {
            bins[r % m] += 1;
        }
        let nonzero: Vec<String> = bins
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, c)| format!("{i}:{c}"))
            .collect();
        println!("{name} dirty mod {m}: [{}]", nonzero.join(" "));
    }
    // Contiguous runs of dirty.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < dirty.len() {
        let start = dirty[i].0;
        let mut end = start;
        let mut j = i + 1;
        while j < dirty.len() && dirty[j].0 == end + 1 {
            end = dirty[j].0;
            j += 1;
        }
        runs.push((start, end));
        i = j;
    }
    print!("{name} dirty runs ({}):", runs.len());
    for (a, b) in runs.iter().take(16) {
        if a == b {
            print!(" [{a}]");
        } else {
            print!(" [{a}..{b}]");
        }
    }
    if runs.len() > 16 {
        print!(" ...+{}", runs.len() - 16);
    }
    println!();
    let _ = per_row;
}

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

fn row_max_abs(a: &[f32], b: &[f32], row: usize, width: usize) -> f64 {
    let aa = &a[row * width..(row + 1) * width];
    let bb = &b[row * width..(row + 1) * width];
    metrics(aa, bb).0
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 4;
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
    let nbytes = nelems * 2;
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

fn upload_f32(gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.upload_f32(data, shape)
        .map_err(|e| format!("deepseek4 parent: upload_f32: {e:?}"))
}
