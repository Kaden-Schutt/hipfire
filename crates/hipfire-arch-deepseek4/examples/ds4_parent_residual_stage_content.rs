// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Dump parent per-stage CONTENT at selected layers for cosine vs ref stages.
//!
//! Walks `parent_layer_forward` internals via public HC/attn/moe helpers and
//! D2Hs after each of:
//!   hc_pre_attn, attn_norm, attn_out, hc_post_attn,
//!   hc_pre_ffn, ffn_norm, moe_out, hc_post_ffn
//!
//! Defaults: layers 0,2 ; positions 0,1,64,400,448,512,800,1023 ; seq 1024.

use hipfire_arch_deepseek4::parent::attention::{
    parent_attention_swa, PARENT_DIM, PARENT_HEAD_DIM, PARENT_N_KV_HEADS, PARENT_RMS_EPS,
    PARENT_SWA_WINDOW,
};
use hipfire_arch_deepseek4::parent::forward::{
    ParentForwardScratch, PARENT_HC_DIM, PARENT_HC_MULT, PARENT_HC_EPS, PARENT_HC_SINKHORN_ITERS,
};
use hipfire_arch_deepseek4::parent::hc::{parent_hc_post, parent_hc_pre, parent_rms_norm, ParentHcParams};
use hipfire_arch_deepseek4::parent::head::parent_embed;
use hipfire_arch_deepseek4::parent::inventory::ParentInventory;
use hipfire_arch_deepseek4::parent::moe::{parent_moe_forward, parent_route};
use hipfire_arch_deepseek4::parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_arch_deepseek4::parent::Ds4ParentBackend;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const DEFAULT_TOKEN_IDS: &str =
    "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin";
const DEFAULT_LAYERS: &[usize] = &[0, 2];
const DEFAULT_POSITIONS: &[usize] = &[0, 1, 64, 400, 448, 512, 800, 1023];

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
    let token_ids = read_token_ids(&args.token_ids)?;
    let n = args.tokens.min(token_ids.len());
    let token_ids = &token_ids[..n];
    let positions: Vec<usize> = args.positions.iter().copied().filter(|&p| p < n).collect();
    let layers = args.layers.clone();
    let last_needed = *layers.iter().max().unwrap_or(&0);
    let end = last_needed + 1;

    println!("=== ds4_parent_residual_stage_content ===");
    println!("layers={layers:?} positions={positions:?} tokens={n}");
    fs::create_dir_all(&args.out_dir).map_err(|e| format!("mkdir: {e}"))?;

    let source = SafetensorsSource::open(model_path)
        .map_err(|e| format!("open: {e}"))?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    if gpu.try_gfx942().is_none() && std::env::var_os("HIPFIRE_DS4_ALLOW_NON_GFX942").is_none() {
        return Err(format!("gfx942 required got {}", gpu.arch));
    }
    let (backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    let inv = ParentInventory::build(&source, &cfg)?;
    let plan = ParentLoadPlan { layers: 0..end.min(cfg.num_hidden_layers), load_experts: true };
    let t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    println!("loaded in {:.3}s", t0.elapsed().as_secs_f64());

    let mut scratch = ParentForwardScratch::new(&mut gpu, &cfg, n)?;
    let hc_a = zeros_f32(&mut gpu, &[n, PARENT_HC_DIM])?;
    let hc_b = zeros_f32(&mut gpu, &[n, PARENT_HC_DIM])?;
    let mut rings = Vec::new();
    for _ in 0..end {
        rings.push(zeros_f32(&mut gpu, &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW])?);
    }
    parent_embed(&mut gpu, backend, &weights, &cfg, token_ids, &hc_a)?;

    let want: std::collections::HashSet<usize> = layers.iter().copied().collect();
    let mut use_a = true;
    let mut dumped = Vec::new();

    for layer_idx in 0..end {
        let (x, out) = if use_a { (&hc_a, &hc_b) } else { (&hc_b, &hc_a) };
        let input_ids = if layer_idx < cfg.num_hash_layers { Some(token_ids) } else { None };
        let layer = &weights.layers[layer_idx - weights.layer_range.start];

        if !want.contains(&layer_idx) {
            // fast path: full layer
            hipfire_arch_deepseek4::parent::forward::parent_layer_forward(
                &mut gpu, backend, &weights, &cfg, &mut scratch, layer_idx, x, n, 0, input_ids, &rings[layer_idx], out,
            )?;
            use_a = !use_a;
            continue;
        }

        let dim = PARENT_DIM;
        let hc = PARENT_HC_MULT;
        let stream_y = scratch.stream_y().sub_offset(0, n * dim);
        let stream_normed = scratch.stream_normed().sub_offset(0, n * dim);
        let stream_block = scratch.stream_block().sub_offset(0, n * dim);
        let residual_hc = scratch.residual_hc().sub_offset(0, n * PARENT_HC_DIM);
        let post = scratch.post().sub_offset(0, n * hc);
        let comb = scratch.comb().sub_offset(0, n * hc * hc);
        let moe_x = scratch.moe_x_bf16().sub_offset(0, n * dim);

        // Attention half
        let attn_hc = ParentHcParams { fn_mat: &layer.hc_attn_fn, base: &layer.hc_attn_base, scale: &layer.hc_attn_scale };
        parent_hc_pre(&mut gpu, backend, x, attn_hc, n, hc, dim, PARENT_RMS_EPS, PARENT_HC_SINKHORN_ITERS, PARENT_HC_EPS, &stream_y, &post, &comb)
            .map_err(|e| format!("hc_pre_attn: {e}"))?;
        dump_pos(&gpu, &args.out_dir, layer_idx, "hc_pre_attn", &stream_y, n, dim, &positions)?;

        parent_rms_norm(&mut gpu, backend, &stream_y, &layer.attn_norm, &stream_normed, n, dim, PARENT_RMS_EPS)
            .map_err(|e| format!("attn_norm: {e}"))?;
        dump_pos(&gpu, &args.out_dir, layer_idx, "attn_norm", &stream_normed, n, dim, &positions)?;

        parent_attention_swa(&mut gpu, backend, layer, &cfg, scratch.attn_scratch_mut(), &stream_normed, n, 0, &rings[layer_idx], &stream_block)
            .map_err(|e| format!("attn: {e}"))?;
        dump_pos(&gpu, &args.out_dir, layer_idx, "attn_out", &stream_block, n, dim, &positions)?;

        parent_hc_post(&mut gpu, backend, &stream_block, x, &post, &comb, n, hc, dim, &residual_hc)
            .map_err(|e| format!("hc_post_attn: {e}"))?;
        dump_pos(&gpu, &args.out_dir, layer_idx, "hc_post_attn", &residual_hc, n, PARENT_HC_DIM, &positions)?;

        // FFN half
        let ffn_hc = ParentHcParams { fn_mat: &layer.hc_ffn_fn, base: &layer.hc_ffn_base, scale: &layer.hc_ffn_scale };
        parent_hc_pre(&mut gpu, backend, &residual_hc, ffn_hc, n, hc, dim, PARENT_RMS_EPS, PARENT_HC_SINKHORN_ITERS, PARENT_HC_EPS, &stream_y, &post, &comb)
            .map_err(|e| format!("hc_pre_ffn: {e}"))?;
        dump_pos(&gpu, &args.out_dir, layer_idx, "hc_pre_ffn", &stream_y, n, dim, &positions)?;

        parent_rms_norm(&mut gpu, backend, &stream_y, &layer.ffn_norm, &stream_normed, n, dim, PARENT_RMS_EPS)
            .map_err(|e| format!("ffn_norm: {e}"))?;
        dump_pos(&gpu, &args.out_dir, layer_idx, "ffn_norm", &stream_normed, n, dim, &positions)?;

        // F32->BF16 stage for MoE (mirror forward.rs)
        stage_f32_to_bf16(&mut gpu, &mut scratch, &stream_normed, n, dim)?;

        let is_hash = layer_idx < cfg.num_hash_layers;
        let routing = parent_route(&mut gpu, backend, layer, &cfg, &moe_x, n, if is_hash { input_ids } else { None })
            .map_err(|e| format!("route: {e}"))?;
        parent_moe_forward(&mut gpu, backend, layer, &cfg, scratch.moe_scratch_mut(), &moe_x, n, &routing, &stream_block)
            .map_err(|e| format!("moe: {e}"))?;
        dump_pos(&gpu, &args.out_dir, layer_idx, "moe_out", &stream_block, n, dim, &positions)?;

        parent_hc_post(&mut gpu, backend, &stream_block, &residual_hc, &post, &comb, n, hc, dim, out)
            .map_err(|e| format!("hc_post_ffn: {e}"))?;
        dump_pos(&gpu, &args.out_dir, layer_idx, "hc_post_ffn", out, n, PARENT_HC_DIM, &positions)?;

        println!("L{layer_idx} stages dumped");
        dumped.push(layer_idx);
        use_a = !use_a;
    }

    let meta = format!(
        "{{\"seq\":{n},\"layers\":{layers:?},\"positions\":{positions:?},\"dumped\":{dumped:?},\"stages\":[\"hc_pre_attn\",\"attn_norm\",\"attn_out\",\"hc_post_attn\",\"hc_pre_ffn\",\"ffn_norm\",\"moe_out\",\"hc_post_ffn\"]}}\n"
    );
    fs::write(args.out_dir.join("residual_stage_content_parent.json"), meta).map_err(|e| e.to_string())?;
    println!("wrote meta");
    Ok(())
}

fn stage_f32_to_bf16(
    gpu: &mut Gpu,
    scratch: &mut ParentForwardScratch,
    src: &GpuTensor,
    rows: usize,
    dim: usize,
) -> Result<(), String> {
    // Use the same path as parent_layer_forward: download f32, cast, upload bf16.
    // ParentForwardScratch has host staging but methods are private; do local cast.
    let ne = rows * dim;
    let host = download_f32(gpu, src, ne)?;
    let mut bytes = vec![0u8; ne * 2];
    for (i, &v) in host.iter().enumerate() {
        let bf = hipfire_arch_deepseek4::parent::codec::round_to_bf16(v);
        let bits = (bf.to_bits() >> 16) as u16;
        let b = bits.to_le_bytes();
        bytes[i * 2] = b[0];
        bytes[i * 2 + 1] = b[1];
    }
    let dst = scratch.moe_x_bf16().sub_offset(0, ne);
    gpu.hip
        .memcpy_htod(&dst.buf, &bytes)
        .map_err(|e| format!("htod bf16: {e:?}"))?;
    Ok(())
}

fn dump_pos(
    gpu: &Gpu,
    out_dir: &Path,
    layer: usize,
    name: &str,
    t: &GpuTensor,
    rows: usize,
    row_dim: usize,
    positions: &[usize],
) -> Result<(), String> {
    let host = download_f32(gpu, t, rows * row_dim)?;
    let mut out = Vec::with_capacity(positions.len() * row_dim);
    for &p in positions {
        let base = p * row_dim;
        out.extend_from_slice(&host[base..base + row_dim]);
    }
    let path = out_dir.join(format!("L{layer}_{name}.f32"));
    write_f32_le(&path, &out)?;
    let g = {
        let mut s = 0.0f64;
        for &v in &out { let d = v as f64; s += d * d; }
        s.sqrt()
    };
    println!("  L{layer}_{name} shape=[{}, {row_dim}] dump_l2={g:.4}", positions.len());
    Ok(())
}

fn write_f32_le(path: &Path, data: &[f32]) -> Result<(), String> {
    let mut f = fs::File::create(path).map_err(|e| format!("{e}"))?;
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data { bytes.extend_from_slice(&v.to_le_bytes()); }
    f.write_all(&bytes).map_err(|e| format!("{e}"))
}

fn zeros_f32(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.alloc_tensor(shape, DType::F32).map_err(|e| format!("alloc {shape:?}: {e:?}"))
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    if t.dtype != DType::F32 { return Err(format!("want F32 got {:?}", t.dtype)); }
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes { return Err("buf short".into()); }
    let mut host = vec![0.0f32; nelems];
    let bytes = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).map_err(|e| format!("dtoh: {e:?}"))?;
    Ok(host)
}

fn read_token_ids(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{e}"))?;
    if bytes.len() % 4 != 0 { return Err("bad token file".into()); }
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())
}

struct Args {
    model: String,
    token_ids: PathBuf,
    tokens: usize,
    layers: Vec<usize>,
    positions: Vec<usize>,
    out_dir: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut model = DEFAULT_MODEL.to_string();
    let mut token_ids = PathBuf::from(DEFAULT_TOKEN_IDS);
    let mut tokens = 1024usize;
    let mut layers = DEFAULT_LAYERS.to_vec();
    let mut positions = DEFAULT_POSITIONS.to_vec();
    let mut out_dir = PathBuf::from("/tmp/residual_stage_content_parent");
    let mut argv = std::env::args().skip(1);
    while let Some(a) = argv.next() {
        match a.as_str() {
            "--model" => model = argv.next().ok_or("--model")?,
            "--token-ids" => token_ids = PathBuf::from(argv.next().ok_or("--token-ids")?),
            "--tokens" => tokens = argv.next().ok_or("--tokens")?.parse().map_err(|e| format!("{e}"))?,
            "--layers" => {
                let s = argv.next().ok_or("--layers")?;
                layers = s.split(',').map(|x| x.parse().map_err(|e| format!("{e}"))).collect::<Result<_,_>>()?;
            }
            "--positions" => {
                let s = argv.next().ok_or("--positions")?;
                positions = s.split(',').map(|x| x.parse().map_err(|e| format!("{e}"))).collect::<Result<_,_>>()?;
            }
            "--out-dir" => out_dir = PathBuf::from(argv.next().ok_or("--out-dir")?),
            other => return Err(format!("unknown {other}")),
        }
    }
    Ok(Args { model, token_ids, tokens, layers, positions, out_dir })
}
