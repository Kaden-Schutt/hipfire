// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Real-residual probe for ratio-4 compressed-slot causality + attn_inv.
//!
//! Feeds layer 2 the actual GPU residual from a 128-token parent forward
//! (layers 0..2), dumps GPU topk_idx / n_active, checks each late row's
//! compressed slots are causally valid, and compares attn_inv vs the joint
//! oracle. Layer 0 and layer 3 are floor-calibrated in the same process.
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_indexer_causality \
//!   -- --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
//!      --token-ids /mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin \
//!      --rows 128
//! ```
use hipfire_arch_deepseek4::parent::attention::{
    all_finite, compress_n_visible, get_compress_topk_idxs, l2_norm, parent_attention_swa,
    swa_n_valid, ParentAttnScratch, PARENT_ATTN_INDEX_TOPK, PARENT_DIM, PARENT_HEAD_DIM,
    PARENT_N_HEADS, PARENT_N_KV_HEADS, PARENT_Q_LORA, PARENT_Q_WIDTH, PARENT_RMS_EPS,
    PARENT_SWA_WINDOW, PARENT_WO_A_OUT,
};
use hipfire_arch_deepseek4::parent::codec::round_to_bf16;
use hipfire_arch_deepseek4::parent::compressor::compressor_prefill_windows;
use hipfire_arch_deepseek4::parent::forward::{
    parent_layer_forward, ParentForwardScratch, PARENT_HC_DIM, PARENT_HC_EPS, PARENT_HC_MULT,
    PARENT_HC_SINKHORN_ITERS,
};
use hipfire_arch_deepseek4::parent::head::parent_embed;
use hipfire_arch_deepseek4::parent::indexer::{
    indexer_n_compressed, indexer_n_visible, PARENT_INDEX_TOPK,
};
use hipfire_arch_deepseek4::parent::inventory::ParentInventory;
use hipfire_arch_deepseek4::parent::layer_ref::{
    attention_swa_ref, hc_pre_ref, rms_norm_ref, AttnCompRefWeights, AttnIndexerRefWeights,
    AttnRefOut, AttnSwARefWeights,
};
use hipfire_arch_deepseek4::parent::weights::{ParentLayerWeights, ParentLoadPlan, ParentWeights};
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
const START_POS: usize = 0;
const FLOOR: f64 = 5e-6;
const LATE_ROWS: &[usize] = &[10, 119, 120, 121, 122, 123, 124, 125, 126, 127];

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

    println!("=== ds4_parent_indexer_causality ===");
    println!("model: {}", model_path.display());
    println!("token_ids: {} (n={rows})", args.token_ids.display());
    println!("start_pos: {START_POS}");
    println!("floor: {FLOOR:.0e}  late_rows: {LATE_ROWS:?}");

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
    for &(li, er) in &[(0usize, 0usize), (2, 4), (3, 128)] {
        let got = cfg.compress_ratio(li);
        if got != er {
            return Err(format!(
                "deepseek4 parent: expected layer {li} compress_ratio={er}, got {got}"
            ));
        }
    }
    let inv = ParentInventory::build(&source, &cfg)?;

    // ── Real residuals: embed + layers 0,1 → layer-2 HC input ───────────
    // Load 0..4 so we can also floor-check L0 and L3 without reloading the
    // whole parent.
    let plan = ParentLoadPlan {
        layers: 0..4,
        load_experts: true,
    };
    let t_load = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    println!(
        "loaded layers 0..4 in {:.2}s  resident={:.3} GiB",
        t_load.elapsed().as_secs_f64(),
        weights.residency().total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let mut scratch = ParentForwardScratch::new(&mut gpu, &cfg, rows)?;
    let hc_a = zeros_f32(&mut gpu, &[rows, PARENT_HC_MULT, PARENT_DIM])?;
    let hc_b = zeros_f32(&mut gpu, &[rows, PARENT_HC_MULT, PARENT_DIM])?;
    let mut kv_rings = Vec::with_capacity(4);
    for i in 0..4 {
        kv_rings.push(
            zeros_f32(
                &mut gpu,
                &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
            )
            .map_err(|e| format!("kv_ring[{i}]: {e}"))?,
        );
    }

    parent_embed(&mut gpu, backend, &weights, &cfg, &token_ids, &hc_a)?;

    // Layer 0 → capture attn_norm stream for floor (post hc_pre residual path
    // is inside parent_layer_forward; we re-run attention on the layer's
    // real attn input by extracting stream_block after a dedicated attn
    // path). Simpler: after each layer, residual_hc is post-attn HC; the
    // attn *input* stream is hc_pre's y. For attention oracle we need the
    // post-attn_norm activation. parent_layer_forward leaves stream_normed
    // as ffn_norm output, not attn_norm. So reconstruct attn input:
    //   hc_pre(x) → y; attn_norm(y) — done inside layer. We capture by
    // re-running hc_pre + attn_norm on host for the layers we probe, OR
    // by using the GPU residual stream just before attention via a thin
    // helper. The bisect does: hc_pre_ref → BF16(attn_norm). Match that.

    // Drive layers 0 and 1 to produce real L2 input HC.
    let mut use_a = true;
    for layer_i in 0..2 {
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
            START_POS,
            input_ids,
            &kv_rings[layer_i],
            out,
        )?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync L{layer_i}: {e:?}"))?;
        use_a = !use_a;
    }
    let l2_hc_in = if use_a { &hc_a } else { &hc_b };
    let l2_hc_host = download_f32(&gpu, l2_hc_in, rows * PARENT_HC_DIM)?;
    println!(
        "real L2 HC input: rows={rows} hc_dim={PARENT_HC_DIM} l2={:.6}",
        l2_norm(&l2_hc_host)
    );

    // Build real attn_norm input for layer 2 from HC via the same host
    // path the bisect uses (hc_pre → attn_norm, BF16 round).
    let layer2 = weights
        .layers
        .iter()
        .find(|l| l.layer_idx == 2)
        .ok_or_else(|| "layer 2 missing".to_owned())?;
    let x_attn_l2 = real_attn_input_from_hc(&gpu, layer2, &cfg, &l2_hc_host, rows)?;
    println!(
        "real L2 attn_norm input: dim={PARENT_DIM} l2={:.6} finite={}",
        l2_norm(&x_attn_l2),
        all_finite(&x_attn_l2)
    );

    // Also build L0 and L3 attn inputs for floor (from embed HC for L0;
    // for L3 drive layers 0..3 first).
    // L0: embed is already in hc_a at start — re-embed into a fresh buffer.
    let hc0 = zeros_f32(&mut gpu, &[rows, PARENT_HC_MULT, PARENT_DIM])?;
    parent_embed(&mut gpu, backend, &weights, &cfg, &token_ids, &hc0)?;
    let l0_hc = download_f32(&gpu, &hc0, rows * PARENT_HC_DIM)?;
    let layer0 = weights
        .layers
        .iter()
        .find(|l| l.layer_idx == 0)
        .ok_or_else(|| "layer 0 missing".to_owned())?;
    let x_attn_l0 = real_attn_input_from_hc(&gpu, layer0, &cfg, &l0_hc, rows)?;

    // ── Probe layers ────────────────────────────────────────────────────
    probe_layer(
        &mut gpu,
        backend,
        &weights,
        &cfg,
        0,
        0,
        &x_attn_l0,
        rows,
        /*dump_topk=*/ false,
    )?;
    probe_layer(
        &mut gpu,
        backend,
        &weights,
        &cfg,
        2,
        4,
        &x_attn_l2,
        rows,
        /*dump_topk=*/ true,
    )?;

    // L3 real residual: continue from L2 HC through L2 then take L3 input.
    // We already have L2 HC input in l2_hc_in; run L2 then capture.
    {
        let (x, out) = if use_a {
            (&hc_a, &hc_b)
        } else {
            (&hc_b, &hc_a)
        };
        // x currently holds L2 input (we didn't mutate hc after L1).
        parent_layer_forward(
            &mut gpu,
            backend,
            &weights,
            &cfg,
            &mut scratch,
            2,
            x,
            rows,
            START_POS,
            None,
            &kv_rings[2],
            out,
        )?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync L2: {e:?}"))?;
        use_a = !use_a;
    }
    let l3_hc_in = if use_a { &hc_a } else { &hc_b };
    let l3_hc_host = download_f32(&gpu, l3_hc_in, rows * PARENT_HC_DIM)?;
    let layer3 = weights
        .layers
        .iter()
        .find(|l| l.layer_idx == 3)
        .ok_or_else(|| "layer 3 missing".to_owned())?;
    let x_attn_l3 = real_attn_input_from_hc(&gpu, layer3, &cfg, &l3_hc_host, rows)?;
    probe_layer(
        &mut gpu,
        backend,
        &weights,
        &cfg,
        3,
        128,
        &x_attn_l3,
        rows,
        /*dump_topk=*/ false,
    )?;

    println!();
    println!("DONE");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn probe_layer(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    weights: &ParentWeights,
    cfg: &ParentQuantConfig,
    layer_idx: usize,
    ratio: usize,
    x_f32: &[f32],
    rows: usize,
    dump_topk: bool,
) -> Result<(), String> {
    println!();
    println!("{}", "#".repeat(72));
    println!("# layer {layer_idx}  compress_ratio={ratio}  REAL residual");
    println!("{}", "#".repeat(72));

    let layer = weights
        .layers
        .iter()
        .find(|l| l.layer_idx == layer_idx)
        .ok_or_else(|| format!("layer {layer_idx} missing"))?;
    if layer.compress_ratio != ratio {
        return Err(format!(
            "layer {layer_idx} compress_ratio={} != {ratio}",
            layer.compress_ratio
        ));
    }

    // Host weights for joint oracle.
    let wq_a = download_bf16_as_f32(gpu, layer.wq_a.tensor(), layer.wq_a.n() * layer.wq_a.k())?;
    let wq_b = download_bf16_as_f32(gpu, layer.wq_b.tensor(), layer.wq_b.n() * layer.wq_b.k())?;
    let wkv = download_bf16_as_f32(gpu, layer.wkv.tensor(), layer.wkv.n() * layer.wkv.k())?;
    let wo_a = download_bf16_as_f32(gpu, layer.wo_a.tensor(), layer.wo_a.n() * layer.wo_a.k())?;
    let wo_b = download_bf16_as_f32(gpu, layer.wo_b.tensor(), layer.wo_b.n() * layer.wo_b.k())?;
    let q_norm = download_bf16_as_f32(gpu, &layer.q_norm, PARENT_Q_LORA)?;
    let kv_norm = download_bf16_as_f32(gpu, &layer.kv_norm, PARENT_HEAD_DIM)?;
    let attn_sink = download_f32(gpu, &layer.attn_sink, PARENT_N_HEADS)?;

    let (comp_wkv, comp_wgate, comp_norm, comp_ape) = if let Some(c) = layer.compressor.as_ref() {
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
    let (ix_wq_b, ix_wp, ix_wkv, ix_wgate, ix_norm, ix_ape) =
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
                Some(download_bf16_as_f32(
                    gpu,
                    &ix.compressor_wgate,
                    cproj * cdim,
                )?),
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
    let n_comp_ref = if PARENT_HEAD_DIM > 0 {
        reference.kv_compress.len() / PARENT_HEAD_DIM
    } else {
        0
    };
    let k_comp_stride = if rows > 0 {
        reference.compress_idxs.len() / rows
    } else {
        0
    };
    println!(
        "oracle done in {:.2}s  o_l2={:.6}  n_comp={n_comp_ref} k_comp_stride={k_comp_stride}",
        t_ref.elapsed().as_secs_f64(),
        l2_norm(&reference.o)
    );

    let mut scratch = ParentAttnScratch::new(gpu, cfg, rows)?;
    let kv_ring = gpu
        .zeros(
            &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
            DType::F32,
        )
        .map_err(|e| format!("kv_ring: {e:?}"))?;
    let x = upload_f32(gpu, x_f32, &[rows, PARENT_DIM])?;
    let out = gpu
        .zeros(&[rows, PARENT_DIM], DType::F32)
        .map_err(|e| format!("out: {e:?}"))?;

    let t_gpu = Instant::now();
    parent_attention_swa(
        gpu, backend, layer, cfg, &mut scratch, &x, rows, START_POS, &kv_ring, &out,
    )?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    println!(
        "gpu forward in {:.2}s  last_compress_events={}",
        t_gpu.elapsed().as_secs_f64(),
        scratch.last_compress_events()
    );

    let gpu_attn_inv = download_f32(gpu, scratch.attn_out_f32_ref()?, rows * PARENT_Q_WIDTH)?;
    let gpu_o = download_f32(gpu, &out, rows * PARENT_DIM)?;
    if !all_finite(&gpu_attn_inv) {
        return Err(format!("layer {layer_idx}: GPU attn_inv non-finite"));
    }

    // ── Full 128-row attn_inv dirty set ─────────────────────────────────
    let mut dirty: Vec<(usize, f64)> = Vec::new();
    let mut global_max = 0.0f64;
    let mut global_argmax = 0usize;
    for r in 0..rows {
        let (mx, _, _) = row_metrics(&gpu_attn_inv, &reference.attn_inv_rope, r, PARENT_Q_WIDTH);
        if mx > global_max {
            global_max = mx;
            global_argmax = r;
        }
        if mx > FLOOR {
            dirty.push((r, mx));
        }
    }
    let clean = rows - dirty.len();
    println!();
    println!("=== attn_inv full 128-row scan (floor={FLOOR:.0e}) ===");
    println!(
        "attn_inv: clean={clean}/{rows}  dirty={}  global_max={global_max:.6e} @row{global_argmax}",
        dirty.len()
    );
    if dirty.is_empty() {
        println!("attn_inv dirty rows: (none)");
    } else {
        print!("attn_inv dirty rows:");
        for (r, mx) in &dirty {
            print!(" {r}:{mx:.3e}");
        }
        println!();
        // runs
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut s = dirty[0].0;
        let mut e = s;
        for &(r, _) in dirty.iter().skip(1) {
            if r == e + 1 {
                e = r;
            } else {
                runs.push((s, e));
                s = r;
                e = r;
            }
        }
        runs.push((s, e));
        print!("attn_inv dirty runs ({}):", runs.len());
        for (a, b) in &runs {
            if a == b {
                print!(" [{a}]");
            } else {
                print!(" [{a}..{b}]");
            }
        }
        println!();
    }

    // Floor-spot rows.
    println!();
    println!("         stage     row       max_abs");
    for &r in &[0usize, 1, 8, 10, 32, 64, 100, 119, 127] {
        if r >= rows {
            continue;
        }
        let (mx, _, _) = row_metrics(&gpu_attn_inv, &reference.attn_inv_rope, r, PARENT_Q_WIDTH);
        println!("  attn_inv  {r:>6}   {mx:.6e}");
    }
    let (omax, _, _) = global_metrics(&gpu_o, &reference.o);
    println!("  final_o global_max_abs={omax:.6e}");

    if dump_topk && ratio == 4 {
        dump_causality(gpu, &scratch, &reference, rows, ratio)?;
    } else if ratio == 128 {
        // Identity path: print n_active + compress plan.
        let n_active = download_i32(gpu, scratch.n_active_topk_ref(), rows)?;
        let n_comp = indexer_n_compressed(START_POS, rows, ratio);
        println!();
        println!("=== ratio-128 identity path ===");
        println!("n_compressed={n_comp}");
        for &r in LATE_ROWS {
            if r >= rows {
                continue;
            }
            let vis = compress_n_visible(START_POS, r, ratio);
            println!(
                "  row {r:>3}: n_active={}  vis={vis}  swa_n_valid={}",
                n_active[r],
                swa_n_valid(START_POS, r, PARENT_SWA_WINDOW)
            );
        }
    }

    Ok(())
}

fn dump_causality(
    gpu: &Gpu,
    scratch: &ParentAttnScratch,
    reference: &AttnRefOut,
    rows: usize,
    ratio: usize,
) -> Result<(), String> {
    let n_comp = indexer_n_compressed(START_POS, rows, ratio);
    let topk = download_i32(gpu, scratch.topk_idx_ref(), rows * PARENT_ATTN_INDEX_TOPK)?;
    let n_active = download_i32(gpu, scratch.n_active_topk_ref(), rows)?;
    let plan = compressor_prefill_windows(rows, ratio)?;

    // Host compress-topk (identity causal) for comparison.
    let host_comp = get_compress_topk_idxs(ratio, rows, START_POS, /*offset=*/ 0)?;
    let host_k = if host_comp.is_empty() {
        0
    } else {
        host_comp.len() / rows
    };

    println!();
    println!("=== compressed causality dump (ratio={ratio}, n_comp={n_comp}) ===");
    println!("PARENT_ATTN_INDEX_TOPK={PARENT_ATTN_INDEX_TOPK}  PARENT_INDEX_TOPK={PARENT_INDEX_TOPK}");
    println!("oracle compress_idxs stride={}  host get_compress_topk k={host_k}", {
        if rows > 0 {
            reference.compress_idxs.len() / rows
        } else {
            0
        }
    });

    // Per-row n_active summary.
    let mut n_active_mismatch = 0usize;
    for r in 0..rows {
        let vis = indexer_n_visible(START_POS, r, ratio).min(n_comp);
        let expect = PARENT_ATTN_INDEX_TOPK.min(vis) as i32;
        if n_active[r] != expect {
            n_active_mismatch += 1;
            if n_active_mismatch <= 8 {
                println!(
                    "  n_active mismatch row {r}: gpu={} expect={expect} vis={vis}",
                    n_active[r]
                );
            }
        }
    }
    println!(
        "n_active vs min(topk,vis): mismatches={n_active_mismatch}/{rows}"
    );

    // Full-row causality + packing scan.
    let mut rows_with_future = 0usize;
    let mut rows_with_hole_in_prefix = 0usize;
    let mut rows_n_valid_ne_n_active = 0usize;
    let mut max_future_slot = -1i32;
    let mut future_examples: Vec<String> = Vec::new();

    for r in 0..rows {
        let na = n_active[r].max(0) as usize;
        let row = &topk[r * PARENT_ATTN_INDEX_TOPK..(r + 1) * PARENT_ATTN_INDEX_TOPK];
        let cutoff = indexer_n_visible(START_POS, r, ratio); // first illegal slot
        let mut n_valid = 0usize;
        let mut saw_neg = false;
        let mut hole = false;
        let mut future: Vec<i32> = Vec::new();
        for (j, &idx) in row.iter().enumerate() {
            if j >= na {
                break;
            }
            if idx < 0 {
                saw_neg = true;
                continue;
            }
            if saw_neg {
                hole = true;
            }
            n_valid += 1;
            let s = idx as usize;
            // Slot s is causally valid iff s < cutoff.
            // Also check token coverage via compressor windows.
            let mut tok_ok = s < cutoff;
            if s < plan.n_out {
                let cur = &plan.current_windows[s];
                let prev = &plan.prev_windows[s];
                let max_tok = cur
                    .iter()
                    .chain(prev.iter())
                    .copied()
                    .max()
                    .unwrap_or(0);
                // Query at row r may see tokens ≤ r.
                if max_tok > r {
                    tok_ok = false;
                }
            }
            if !tok_ok || s >= cutoff {
                future.push(idx);
                if idx > max_future_slot {
                    max_future_slot = idx;
                }
            }
        }
        if !future.is_empty() {
            rows_with_future += 1;
            if future_examples.len() < 6 {
                future_examples.push(format!("row {r}: future_slots={future:?} cutoff={cutoff}"));
            }
        }
        if hole {
            rows_with_hole_in_prefix += 1;
        }
        if n_valid != na {
            // -1 inside prefix means n_valid < n_active (padding not packed).
            rows_n_valid_ne_n_active += 1;
        }
    }
    println!(
        "causality: rows_with_future_slots={rows_with_future}/{rows}  max_future_slot={max_future_slot}"
    );
    println!(
        "packing: rows_with_-1_hole_in_n_active_prefix={rows_with_hole_in_prefix}/{rows}"
    );
    println!(
        "packing: rows_n_valid_in_prefix != n_active={rows_n_valid_ne_n_active}/{rows}"
    );
    for e in &future_examples {
        println!("  {e}");
    }

    // Explicit late-row dump.
    println!();
    println!("=== late-row compressed slot lists ===");
    let oracle_k = if rows > 0 {
        reference.compress_idxs.len() / rows
    } else {
        0
    };
    for &r in LATE_ROWS {
        if r >= rows {
            continue;
        }
        let na = n_active[r].max(0) as usize;
        let cutoff = indexer_n_visible(START_POS, r, ratio);
        let row = &topk[r * PARENT_ATTN_INDEX_TOPK..(r + 1) * PARENT_ATTN_INDEX_TOPK];
        let mut gpu_slots: Vec<i32> = Vec::new();
        let mut gpu_valid: Vec<(i32, bool, String)> = Vec::new();
        for &idx in row.iter().take(na.max(32).min(PARENT_ATTN_INDEX_TOPK)) {
            gpu_slots.push(idx);
        }
        for &idx in row.iter().take(na) {
            if idx < 0 {
                gpu_valid.push((idx, false, "-1 pad".into()));
                continue;
            }
            let s = idx as usize;
            let causal = s < cutoff;
            let mut detail = format!("slot {s} cutoff={cutoff}");
            if s < plan.n_out {
                let cur = &plan.current_windows[s];
                let prev = &plan.prev_windows[s];
                let max_tok = cur
                    .iter()
                    .chain(prev.iter())
                    .copied()
                    .max()
                    .map(|t| t as i32)
                    .unwrap_or(-1);
                detail.push_str(&format!(
                    " cur={cur:?} prev={prev:?} max_tok={max_tok} q_row={r}"
                ));
                if max_tok > r as i32 {
                    detail.push_str(" FUTURE_TOKEN");
                }
            }
            gpu_valid.push((idx, causal, detail));
        }
        // Oracle compress idxs for this row (offset=rows in unified space).
        let mut ora: Vec<i32> = Vec::new();
        if oracle_k > 0 {
            for &v in &reference.compress_idxs[r * oracle_k..(r + 1) * oracle_k] {
                // Strip unified offset to compressed-local.
                if v < 0 {
                    ora.push(-1);
                } else {
                    ora.push(v - rows as i32);
                }
            }
        }
        let mut host: Vec<i32> = Vec::new();
        if host_k > 0 {
            host.extend_from_slice(&host_comp[r * host_k..(r + 1) * host_k]);
        }

        println!(
            "row {r}: n_active={na} vis_cutoff={cutoff} swa_n_valid={}",
            swa_n_valid(START_POS, r, PARENT_SWA_WINDOW)
        );
        println!("  gpu topk[0..max(na,32)] = {:?}", &gpu_slots);
        println!("  oracle compress (local) = {:?}", &ora[..ora.len().min(40)]);
        println!("  host get_compress_topk  = {:?}", &host[..host.len().min(40)]);
        let mut n_bad = 0usize;
        for (idx, ok, detail) in &gpu_valid {
            if !ok {
                n_bad += 1;
                println!("  BAD idx={idx}  {detail}");
            }
        }
        if n_bad == 0 {
            println!("  all {na} active slots causally valid");
        } else {
            println!("  {n_bad} causally INVALID slots in n_active prefix");
        }

        // Set compare gpu vs oracle (ignore -1, ignore order).
        let gset: std::collections::BTreeSet<i32> = row
            .iter()
            .copied()
            .filter(|&v| v >= 0)
            .take(na)
            .collect();
        let oset: std::collections::BTreeSet<i32> = ora.iter().copied().filter(|&v| v >= 0).collect();
        let only_g: Vec<i32> = gset.difference(&oset).copied().collect();
        let only_o: Vec<i32> = oset.difference(&gset).copied().collect();
        if only_g.is_empty() && only_o.is_empty() {
            println!("  set(gpu)==set(oracle)  (|S|={})", gset.len());
        } else {
            println!(
                "  set mismatch: only_gpu={only_g:?} only_oracle={only_o:?} |g|={} |o|={}",
                gset.len(),
                oset.len()
            );
        }
    }

    // Packing contract: for identity fast-path, topk prefix should be
    // 0..vis-1 then -1. Report if any late row deviates.
    println!();
    println!("=== identity-prefix check (n_comp={n_comp} ≤ topk → expect arange) ===");
    for &r in LATE_ROWS {
        if r >= rows {
            continue;
        }
        let na = n_active[r].max(0) as usize;
        let row = &topk[r * PARENT_ATTN_INDEX_TOPK..(r + 1) * PARENT_ATTN_INDEX_TOPK];
        let mut expect_ok = true;
        for j in 0..na {
            if row[j] != j as i32 {
                expect_ok = false;
                break;
            }
        }
        // Beyond n_active, either -1 or future-masked -1 from full identity.
        println!(
            "  row {r}: n_active={na} prefix_is_arange={expect_ok}  first8={:?}",
            &row[..8.min(row.len())]
        );
    }

    Ok(())
}

/// hc_pre → attn_norm, BF16-rounded (bisect domain).
fn real_attn_input_from_hc(
    gpu: &Gpu,
    layer: &ParentLayerWeights,
    _cfg: &ParentQuantConfig,
    hc: &[f32],
    rows: usize,
) -> Result<Vec<f32>, String> {
    let mix_hc = (2 + PARENT_HC_MULT) * PARENT_HC_MULT;
    let hc_flat = PARENT_HC_DIM;
    let hc_fn = download_f32(gpu, &layer.hc_attn_fn, mix_hc * hc_flat)?;
    let hc_base = download_f32(gpu, &layer.hc_attn_base, mix_hc)?;
    let hc_scale = download_f32(gpu, &layer.hc_attn_scale, 3)?;
    let (y, _post, _comb) = hc_pre_ref(
        hc,
        &hc_fn,
        &hc_scale,
        &hc_base,
        rows,
        PARENT_HC_MULT,
        PARENT_DIM,
        PARENT_RMS_EPS as f64,
        PARENT_HC_SINKHORN_ITERS as usize,
        PARENT_HC_EPS as f64,
    )?;
    let attn_norm_w = download_bf16_as_f32(gpu, &layer.attn_norm, PARENT_DIM)?;
    let normed = rms_norm_ref(&y, &attn_norm_w, PARENT_RMS_EPS as f64, PARENT_DIM);
    Ok(normed.iter().map(|&v| round_to_bf16(v)).collect())
}

fn row_metrics(a: &[f32], b: &[f32], row: usize, width: usize) -> (f64, f64, f64) {
    let base = row * width;
    let mut max_abs = 0.0f64;
    let mut sum_rel = 0.0f64;
    let mut n_rel = 0usize;
    let mut sum_sq = 0.0f64;
    let mut sum_b = 0.0f64;
    for i in 0..width {
        let da = a[base + i] as f64;
        let db = b[base + i] as f64;
        let d = (da - db).abs();
        if d > max_abs {
            max_abs = d;
        }
        let denom = db.abs().max(1e-12);
        sum_rel += d / denom;
        n_rel += 1;
        sum_sq += d * d;
        sum_b += db * db;
    }
    let mean_rel = if n_rel > 0 {
        sum_rel / n_rel as f64
    } else {
        0.0
    };
    let l2_rel = if sum_b > 0.0 {
        sum_sq.sqrt() / sum_b.sqrt()
    } else {
        0.0
    };
    (max_abs, mean_rel, l2_rel)
}

fn global_metrics(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f64;
    let mut sum_rel = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum_b = 0.0f64;
    for i in 0..a.len() {
        let da = a[i] as f64;
        let db = b[i] as f64;
        let d = (da - db).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_rel += d / db.abs().max(1e-12);
        sum_sq += d * d;
        sum_b += db * db;
    }
    let n = a.len().max(1) as f64;
    (
        max_abs,
        sum_rel / n,
        if sum_b > 0.0 {
            sum_sq.sqrt() / sum_b.sqrt()
        } else {
            0.0
        },
    )
}

fn read_token_ids(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "token-ids file {} length {} not multiple of 4",
            path.display(),
            bytes.len()
        ));
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
                model = it
                    .next()
                    .ok_or_else(|| "--model needs value".to_owned())?;
            }
            "--token-ids" => {
                token_ids = PathBuf::from(
                    it.next()
                        .ok_or_else(|| "--token-ids needs value".to_owned())?,
                );
            }
            "--rows" => {
                rows = it
                    .next()
                    .ok_or_else(|| "--rows needs value".to_owned())?
                    .parse()
                    .map_err(|e| format!("--rows: {e}"))?;
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: ds4_parent_indexer_causality [--model DIR] [--token-ids PATH] [--rows N]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(Args {
        model,
        token_ids,
        rows,
    })
}

fn zeros_f32(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.zeros(shape, DType::F32)
        .map_err(|e| format!("zeros_f32: {e:?}"))
}

fn upload_f32(gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.upload_f32(data, shape)
        .map_err(|e| format!("upload_f32: {e:?}"))
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes {
        return Err(format!(
            "download_f32: buf {} < need {nbytes}",
            t.buf.size()
        ));
    }
    let mut bytes = vec![0u8; nbytes];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .map_err(|e| format!("download_f32 dtoh: {e:?}"))?;
    let mut out = vec![0f32; nelems];
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
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes {
        return Err(format!(
            "download_i32: buf {} < need {nbytes}",
            t.buf.size()
        ));
    }
    let mut bytes = vec![0u8; nbytes];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .map_err(|e| format!("download_i32 dtoh: {e:?}"))?;
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

fn download_bf16_as_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 2;
    if t.buf.size() < nbytes {
        return Err(format!(
            "download_bf16: buf {} < need {nbytes}",
            t.buf.size()
        ));
    }
    let mut bytes = vec![0u8; nbytes];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .map_err(|e| format!("download_bf16 dtoh: {e:?}"))?;
    let mut out = vec![0f32; nelems];
    for i in 0..nelems {
        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        out[i] = bf16_to_f32(bits);
    }
    Ok(out)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
