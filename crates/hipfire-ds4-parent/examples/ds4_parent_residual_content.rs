// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Dump parent HC residual **content** at selected layers/positions for
//! cosine comparison against `reference_oracle/residual_content_dump.py`.
//!
//! Sibling of the LE-norm path in `ds4_parent_forward_gate` / ParentPosTraj.
//! Does **not** touch `parent/model.rs` — drives layers itself so residual
//! tensors can be D2H'd at chosen layers without expanding the production API.
//!
//! Dumps (defaults match the torch sibling):
//!   layers:    -1(embed), 0, 2, 10, 20, 30, 38, 42
//!   positions: 0, 1, 64, 200, 400, 448, 512, 600, 800, 1000, 1023
//!   layout:    [n_pos, hc_mult=4, dim=4096] f32 LE flat
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_residual_content -- \\
//!   --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \\
//!   --token-ids .../tokens.bin --tokens 1024 \\
//!   --out-dir /tmp/residual_content_parent
//! ```
//!
//! Must run on gfx942 (mi300x). Parent needs ~151 GiB — cannot coexist with
//! CtxScan's MQ2R residency.

use hipfire_ds4_parent::attention::{
    PARENT_DIM, PARENT_HEAD_DIM, PARENT_N_KV_HEADS, PARENT_SWA_WINDOW,
};
use hipfire_ds4_parent::forward::{
    parent_layer_forward, parent_layer_forward_traced, ParentForwardScratch,
    ParentLayerTrace, PARENT_HC_DIM, PARENT_HC_MULT,
};
use hipfire_ds4_parent::head::parent_embed;
use hipfire_ds4_parent::inventory::ParentInventory;
use hipfire_ds4_parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_ds4_parent::Ds4ParentBackend;
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
const DEFAULT_LAYERS: &[i32] = &[-1, 0, 2, 10, 20, 30, 38, 42];
const DEFAULT_POSITIONS: &[usize] = &[0, 1, 64, 200, 400, 448, 512, 600, 800, 1000, 1023];

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
            "--model must be a safetensors directory, got {}",
            model_path.display()
        ));
    }

    let token_ids = read_token_ids(&args.token_ids)?;
    let n = args.tokens.min(token_ids.len());
    if n == 0 {
        return Err("token-ids empty".into());
    }
    let token_ids = &token_ids[..n];

    let positions: Vec<usize> = args
        .positions
        .iter()
        .copied()
        .filter(|&p| p < n)
        .collect();
    if positions.is_empty() {
        return Err("no positions in range".into());
    }

    let layers: Vec<i32> = args.layers.clone();
    let max_layer = layers
        .iter()
        .copied()
        .filter(|&l| l >= 0)
        .max()
        .unwrap_or(-1);
    let last_needed = if max_layer < 0 {
        0usize
    } else {
        max_layer as usize
    };

    println!("=== ds4_parent_residual_content ===");
    println!("model: {}", model_path.display());
    println!("token-ids: {}", args.token_ids.display());
    println!("tokens: {n}");
    println!("layers: {:?}", layers);
    println!("positions: {:?}", positions);
    println!("out-dir: {}", args.out_dir.display());

    fs::create_dir_all(&args.out_dir).map_err(|e| format!("mkdir: {e}"))?;

    let source = SafetensorsSource::open(model_path)
        .map_err(|e| format!("SafetensorsSource::open({}): {e}", model_path.display()))?;

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
        "admit OK: layers={} hash_layers={}",
        cfg.num_hidden_layers, cfg.num_hash_layers
    );

    let inv = ParentInventory::build(&source, &cfg)?;
    let end = (last_needed + 1).min(cfg.num_hidden_layers);
    let plan = ParentLoadPlan {
        layers: 0..end,
        load_experts: true,
    };
    let load_t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    println!(
        "loaded layers 0..{end} in {:.3} s",
        load_t0.elapsed().as_secs_f64()
    );

    let mut layer_scratch = ParentForwardScratch::new(&mut gpu, &cfg, n)?;
    let hc_a = zeros_f32(&mut gpu, &[n, PARENT_HC_DIM])?;
    let hc_b = zeros_f32(&mut gpu, &[n, PARENT_HC_DIM])?;
    let mut rings: Vec<GpuTensor> = Vec::with_capacity(end);
    for _ in 0..end {
        rings.push(zeros_f32(
            &mut gpu,
            &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
        )?);
    }

    let mut dumped: Vec<String> = Vec::new();
    let want: std::collections::HashSet<i32> = layers.iter().copied().collect();

    let emb_t0 = Instant::now();
    parent_embed(&mut gpu, backend, &weights, &cfg, token_ids, &hc_a)?;
    println!("embed done in {:.3} s", emb_t0.elapsed().as_secs_f64());

    if want.contains(&-1) {
        let host = download_f32(&gpu, &hc_a, n * PARENT_HC_DIM)?;
        let path = dump_layer_positions(
            &args.out_dir,
            "layer_-1_embed",
            &host,
            n,
            &positions,
        )?;
        let g = l2_all(&host);
        println!("  dumped embed -> {} global_L2={g:.6}", path.display());
        dumped.push(format!(
            "{{\"layer\":-1,\"name\":\"embed\",\"file\":\"{}\",\"global_l2\":{g}}}",
            path.file_name().and_then(|s| s.to_str()).unwrap_or("")
        ));
    }

    let mut use_a = true;
    let fwd_t0 = Instant::now();
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
        let t0 = Instant::now();
        let want_stage = args.stage_layers.contains(&layer_idx);
        if want_stage {
            let mut trace = ParentLayerTrace::default();
            parent_layer_forward_traced(
                &mut gpu,
                backend,
                &weights,
                &cfg,
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
            // Stage tensors live in scratch; dump selected positions for each stage.
            // stream stages: [rows, dim]; hc stages: [rows, hc_dim]
            dump_stage(
                &gpu,
                &args.out_dir,
                layer_idx,
                "hc_pre_attn",
                layer_scratch.stream_y(),
                n,
                PARENT_DIM,
                &positions,
            )?;
            // attn_norm overwrote stream_normed; attn_out in stream_block; but after full
            // layer these are the FFN-half finals. Traced path only keeps final tiles.
            // Re-dump what is still valid at end: hc_post_ffn is `out`.
            // For intermediate stages we need mid-forward captures — add via
            // downloading during a custom path. Here dump end-state diagnostics:
            dump_stage(
                &gpu,
                &args.out_dir,
                layer_idx,
                "hc_post_ffn",
                out,
                n,
                PARENT_HC_DIM,
                &positions,
            )?;
            println!(
                "  STAGE L{layer_idx} norms hc_pre_attn={:.4} attn_norm={:.4} attn_out={:.4} hc_post_attn={:.4} hc_pre_ffn={:.4} ffn_norm={:.4} moe_out={:.4} hc_post_ffn={:.4}",
                trace.hc_pre_attn, trace.attn_norm, trace.attn_out, trace.hc_post_attn,
                trace.hc_pre_ffn, trace.ffn_norm, trace.moe_out, trace.hc_post_ffn
            );
        } else {
            parent_layer_forward(
                &mut gpu,
                backend,
                &weights,
                &cfg,
                &mut layer_scratch,
                layer_idx,
                x,
                n,
                /*start_pos=*/ 0,
                input_ids,
                &rings[layer_idx],
                out,
            )?;
        }
        let dt = t0.elapsed().as_secs_f64();

        if want.contains(&(layer_idx as i32)) {
            let host = download_f32(&gpu, out, n * PARENT_HC_DIM)?;
            let name = format!("layer_{layer_idx}");
            let path = dump_layer_positions(&args.out_dir, &name, &host, n, &positions)?;
            let g = l2_all(&host);
            let row0 = row_l2(&host, 0);
            let row_last = row_l2(&host, n - 1);
            println!(
                "L{layer_idx:02} fwd={dt:.2}s global_L2={g:.4} p0={row0:.2} p_last={row_last:.2} -> {}",
                path.display()
            );
            dumped.push(format!(
                "{{\"layer\":{layer_idx},\"name\":\"{name}\",\"file\":\"{}\",\"global_l2\":{g},\"fwd_s\":{dt}}}",
                path.file_name().and_then(|s| s.to_str()).unwrap_or("")
            ));
        } else if layer_idx % 5 == 0 || layer_idx + 1 == end {
            println!("L{layer_idx:02} fwd={dt:.2}s (no dump)");
        }

        use_a = !use_a;
    }
    println!("layers done in {:.3} s", fwd_t0.elapsed().as_secs_f64());

    // Manual JSON (avoid serde_json dep in example if not already linked — crate has it).
    let meta = format!(
        r#"{{
  "seq": {n},
  "layers": {layers:?},
  "positions": {positions:?},
  "hc_mult": {hc_mult},
  "dim": {dim},
  "hc_dim": {hc_dim},
  "dtype": "f32",
  "layout": "[n_pos, hc_mult, dim]",
  "endian": "le",
  "tokens_path": "{tokens}",
  "model": "{model}",
  "dumped": [
    {dumped}
  ],
  "notes": [
    "Parent HC residual content after each captured layer (embed as layer=-1)",
    "Arithmetic domain: parent internal F32 residual; weights mixed fp8/fp4/bf16",
    "Binary files are raw f32 LE, shape [n_pos, 4, 4096] row-major"
  ]
}}
"#,
        n = n,
        layers = layers,
        positions = positions,
        hc_mult = PARENT_HC_MULT,
        dim = PARENT_DIM,
        hc_dim = PARENT_HC_DIM,
        tokens = args.token_ids.display(),
        model = args.model,
        dumped = dumped.join(",\n    "),
    );
    let meta_path = args.out_dir.join("residual_content_parent.json");
    fs::write(&meta_path, meta).map_err(|e| format!("write meta: {e}"))?;
    println!("wrote {}", meta_path.display());
    Ok(())
}

fn dump_stage(
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
    println!("  stage dump {} elems={} row_dim={row_dim}", path.display(), out.len());
    Ok(())
}

fn dump_layer_positions(
    out_dir: &Path,
    name: &str,
    host: &[f32],
    rows: usize,
    positions: &[usize],
) -> Result<PathBuf, String> {
    let hc_dim = PARENT_HC_DIM;
    if host.len() < rows * hc_dim {
        return Err(format!(
            "host too short for {name}: {} < {}",
            host.len(),
            rows * hc_dim
        ));
    }
    let mut out = Vec::with_capacity(positions.len() * hc_dim);
    for &p in positions {
        let base = p * hc_dim;
        out.extend_from_slice(&host[base..base + hc_dim]);
    }
    let path = out_dir.join(format!("{name}.f32"));
    write_f32_le(&path, &out)?;
    let mut row_l2s = Vec::with_capacity(rows);
    for r in 0..rows {
        row_l2s.push(row_l2(host, r) as f32);
    }
    write_f32_le(&out_dir.join(format!("{name}_row_l2.f32")), &row_l2s)?;
    Ok(path)
}

fn write_f32_le(path: &Path, data: &[f32]) -> Result<(), String> {
    let mut f = fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&bytes)
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(())
}

fn row_l2(host: &[f32], row: usize) -> f64 {
    let base = row * PARENT_HC_DIM;
    let mut s = 0.0f64;
    for j in 0..PARENT_HC_DIM {
        let v = host[base + j] as f64;
        s += v * v;
    }
    s.sqrt()
}

fn l2_all(host: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &v in host {
        let d = v as f64;
        s += d * d;
    }
    s.sqrt()
}

fn zeros_f32(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.alloc_tensor(shape, DType::F32)
        .map_err(|e| format!("alloc {shape:?}: {e:?}"))
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    if t.dtype != DType::F32 {
        return Err(format!("download expects F32 got {:?}", t.dtype));
    }
    let nbytes = nelems.checked_mul(4).ok_or("size overflow")?;
    if t.buf.size() < nbytes {
        return Err(format!(
            "buffer too small have {} need {nbytes}",
            t.buf.size()
        ));
    }
    let mut host = vec![0.0f32; nelems];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("dtoh: {e:?}"))?;
    Ok(host)
}

fn read_token_ids(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "token-ids {}: length {} not multiple of 4",
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
    tokens: usize,
    layers: Vec<i32>,
    positions: Vec<usize>,
    out_dir: PathBuf,
    stage_layers: Vec<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut model = DEFAULT_MODEL.to_string();
    let mut token_ids = PathBuf::from(DEFAULT_TOKEN_IDS);
    let mut tokens = 1024usize;
    let mut layers: Vec<i32> = DEFAULT_LAYERS.to_vec();
    let mut positions: Vec<usize> = DEFAULT_POSITIONS.to_vec();
    let mut out_dir = PathBuf::from("/tmp/residual_content_parent");
    let mut stage_layers: Vec<usize> = Vec::new();

    let mut argv = std::env::args().skip(1);
    while let Some(a) = argv.next() {
        match a.as_str() {
            "--model" => {
                model = argv.next().ok_or("--model needs value")?;
            }
            "--token-ids" => {
                token_ids = PathBuf::from(argv.next().ok_or("--token-ids needs value")?);
            }
            "--tokens" => {
                tokens = argv
                    .next()
                    .ok_or("--tokens needs value")?
                    .parse()
                    .map_err(|e| format!("--tokens: {e}"))?;
            }
            "--layers" => {
                let s = argv.next().ok_or("--layers needs value")?;
                layers = s
                    .split(',')
                    .map(|x| x.parse::<i32>().map_err(|e| format!("layer {x}: {e}")))
                    .collect::<Result<_, _>>()?;
            }
            "--positions" => {
                let s = argv.next().ok_or("--positions needs value")?;
                positions = s
                    .split(',')
                    .map(|x| x.parse::<usize>().map_err(|e| format!("pos {x}: {e}")))
                    .collect::<Result<_, _>>()?;
            }
            "--out-dir" => {
                out_dir = PathBuf::from(argv.next().ok_or("--out-dir needs value")?);
            }
            "--stage-layers" => {
                let s = argv.next().ok_or("--stage-layers needs value")?;
                stage_layers = s
                    .split(',')
                    .filter(|x| !x.is_empty())
                    .map(|x| x.parse::<usize>().map_err(|e| format!("stage layer {x}: {e}")))
                    .collect::<Result<_, _>>()?;
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: ds4_parent_residual_content --model DIR --token-ids BIN \\\n  [--tokens N] [--layers -1,0,2,...] [--positions 0,1,...] [--out-dir DIR]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    Ok(Args {
        model,
        token_ids,
        tokens,
        layers,
        positions,
        out_dir,
        stage_layers,
    })
}
