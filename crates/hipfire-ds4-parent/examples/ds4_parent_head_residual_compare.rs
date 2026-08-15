// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compare GPU `parent_head` against torch-teacher head stages on the same
//! L42 residual dumps (host-side floor already identity — this catches GPU
//! path defects: hc_head kernel, rmsnorm, BF16 MFMA head GEMM).
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_head_residual_compare \\\n//!   -- --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \\\n//!      --ref-residual /tmp/residual_content_ref/residual_content_ref.npz \\\n//!      --parent-residual /tmp/residual_content_parent/layer_42.f32 \\\n//!      --torch-stages /tmp/head_path_content/head_path_stages.npz \\\n//!      --out /tmp/head_path_content/gpu_parent_head_compare.json
//! ```

use hipfire_ds4_parent::attention::PARENT_DIM;
use hipfire_ds4_parent::head::{
    parent_head, PARENT_HC_DIM, PARENT_HC_MULT, PARENT_VOCAB,
};
use hipfire_ds4_parent::inventory::ParentInventory;
use hipfire_ds4_parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_ds4_parent::Ds4ParentBackend;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const DEFAULT_POSITIONS: [usize; 11] = [0, 1, 64, 200, 400, 448, 512, 600, 800, 1000, 1023];

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
    println!("=== ds4_parent_head_residual_compare ===");
    println!("model: {}", args.model.display());

    let n_pos = DEFAULT_POSITIONS.len();
    let parent_res = read_f32_bin(&args.parent_residual, n_pos * PARENT_HC_DIM)?;
    println!(
        "parent residual: {} values ({n_pos} x {PARENT_HC_MULT} x {PARENT_DIM})",
        parent_res.len()
    );

    // Optional torch stages for direct compare
    let torch = if let Some(p) = args.torch_stages.as_ref() {
        Some(load_torch_stages(p, n_pos)?)
    } else {
        None
    };

    let source = SafetensorsSource::open(&args.model)
        .map_err(|e| format!("SafetensorsSource::open: {e}"))?;
    let mut gpu = Gpu::init().map_err(|e| format!("Gpu::init: {e:?}"))?;
    if gpu.try_gfx942().is_none() {
        return Err("gfx942 required".into());
    }
    let (backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    let inv = ParentInventory::build(&source, &cfg)?;
    let plan = ParentLoadPlan {
        layers: 0..0,
        load_experts: false,
    };
    let t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    println!(
        "globals loaded in {:.2}s resident≈{:.2} GiB",
        t0.elapsed().as_secs_f64(),
        weights.residency().total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Upload parent residual as F32 [n_pos, hc_mult, dim] flat
    let x = gpu
        .alloc_tensor(&[n_pos, PARENT_HC_MULT, PARENT_DIM], DType::F32)
        .map_err(|e| format!("alloc x: {e:?}"))?;
    upload_f32(&mut gpu, &x, &parent_res)?;

    let logits = gpu
        .alloc_tensor(&[n_pos, PARENT_VOCAB], DType::F32)
        .map_err(|e| format!("alloc logits: {e:?}"))?;

    let t1 = Instant::now();
    parent_head(
        &mut gpu,
        backend,
        &weights,
        &cfg,
        &x,
        n_pos,
        &logits,
    )?;
    let head_ms = t1.elapsed().as_secs_f64() * 1e3;
    println!("parent_head {n_pos} rows: {head_ms:.1} ms");

    let gpu_logits = gpu
        .download_f32(&logits)
        .map_err(|e| format!("download logits: {e:?}"))?;

    // Also run on REF residual if we can load it from npz via pre-extracted bin
    let mut report = serde_json::json!({
        "n_pos": n_pos,
        "positions": DEFAULT_POSITIONS.to_vec(),
        "parent_head_ms": head_ms,
    });

    if let Some(ref tr) = torch {
        // Compare GPU logits vs torch logits on PARENT residual
        let m = metrics(&gpu_logits, &tr.parent_logits_torch);
        println!(
            "GPU parent_head(parent_res) vs torch_head(parent_res): cos={:.8} rel={:.6e} nr={:.6}",
            m.0, m.1, m.2
        );
        report["gpu_vs_torch_on_parent_residual"] = serde_json::json!({
            "cosine": m.0, "rel_l2": m.1, "norm_ratio": m.2
        });

        // per-pos
        let mut rows = Vec::new();
        for (i, &p) in DEFAULT_POSITIONS.iter().enumerate() {
            let a = &gpu_logits[i * PARENT_VOCAB..(i + 1) * PARENT_VOCAB];
            let b = &tr.parent_logits_torch[i * PARENT_VOCAB..(i + 1) * PARENT_VOCAB];
            let mm = metrics(a, b);
            // top1 tokens
            let ta = argmax(a);
            let tb = argmax(b);
            rows.push(serde_json::json!({
                "pos": p,
                "cosine": mm.0,
                "rel_l2": mm.1,
                "norm_ratio": mm.2,
                "top1_gpu": ta,
                "top1_torch": tb,
                "top1_agree": ta == tb,
            }));
            println!(
                "  pos={p:4} cos={:.8} rel={:.4e} top1_gpu={ta} top1_torch={tb} agree={}",
                mm.0, mm.1, ta == tb
            );
        }
        report["per_pos_gpu_vs_torch_parent_res"] = serde_json::Value::Array(rows);

        // GPU vs torch on REF residual if available
        if let Some(ref_res_path) = args.ref_residual_bin.as_ref() {
            let ref_res = read_f32_bin(ref_res_path, n_pos * PARENT_HC_DIM)?;
            upload_f32(&mut gpu, &x, &ref_res)?;
            parent_head(&mut gpu, backend, &weights, &cfg, &x, n_pos, &logits)?;
            let gpu_ref = gpu.download_f32(&logits).map_err(|e| format!("{e:?}"))?;
            let m2 = metrics(&gpu_ref, &tr.ref_logits_torch);
            println!(
                "GPU parent_head(ref_res) vs torch_head(ref_res): cos={:.8} rel={:.6e} nr={:.6}",
                m2.0, m2.1, m2.2
            );
            report["gpu_vs_torch_on_ref_residual"] = serde_json::json!({
                "cosine": m2.0, "rel_l2": m2.1, "norm_ratio": m2.2
            });
            let mut agree = 0usize;
            for i in 0..n_pos {
                let a = &gpu_ref[i * PARENT_VOCAB..(i + 1) * PARENT_VOCAB];
                let b = &tr.ref_logits_torch[i * PARENT_VOCAB..(i + 1) * PARENT_VOCAB];
                if argmax(a) == argmax(b) {
                    agree += 1;
                }
            }
            report["top1_agree_on_ref_residual"] = serde_json::json!(agree as f64 / n_pos as f64);
            println!("top1 agree on ref residual: {agree}/{n_pos}");
        }

        // Host floor already known; if GPU vs torch on ref residual is near 1,
        // head GPU path is clean.
        let floor = report
            .get("gpu_vs_torch_on_ref_residual")
            .and_then(|v| v.get("cosine"))
            .and_then(|v| v.as_f64())
            .unwrap_or(m.0);
        let (verdict, msg) = if floor >= 0.9999 {
            (
                "GPU_HEAD_AT_FLOOR",
                format!(
                    "GPU parent_head matches torch head on identical residual (cos={floor:.8}). \
                     Head path is not the 12.7x PPL bug; residual content / full-seq path remains."
                ),
            )
        } else if floor >= 0.999 {
            (
                "GPU_HEAD_NEAR_FLOOR",
                format!("GPU parent_head cos={floor:.8} near floor — minor GEMM noise"),
            )
        } else {
            (
                "GPU_HEAD_DEFECT",
                format!(
                    "GPU parent_head cos={floor:.8} well below identity on identical residual — head port bug"
                ),
            )
        };
        report["verdict"] = serde_json::json!(verdict);
        report["verdict_msg"] = serde_json::json!(msg);
        println!("VERDICT: {verdict}");
        println!("{msg}");
    } else {
        println!("no --torch-stages; wrote GPU logits only stats");
        report["verdict"] = serde_json::json!("NO_TORCH_STAGES");
    }

    // free
    let _ = gpu.free_tensor(x);
    let _ = gpu.free_tensor(logits);
    let _keep = (weights, backend, cfg);
    if let Some(out) = args.out.as_ref() {
        std::fs::write(out, serde_json::to_string_pretty(&report).unwrap())
            .map_err(|e| format!("write {out}: {e}"))?;
        println!("wrote {out}");
    }
    Ok(())
}

struct TorchStages {
    parent_logits_torch: Vec<f32>,
    ref_logits_torch: Vec<f32>,
}

fn load_torch_stages(path: &Path, n_pos: usize) -> Result<TorchStages, String> {
    // Minimal NPZ reader for float32 arrays we need. Prefer npy crate-less: shell out? 
    // Use a tiny pure-rust npz via flate2+zip if available, else require pre-extracted bins.
    // Simpler: call python to dump bins next to npz.
    let dir = path.parent().unwrap_or(Path::new("."));
    let pref = dir.join("_gpu_cmp_parent_logits_torch.f32");
    let rref = dir.join("_gpu_cmp_ref_logits_torch.f32");
    if !(pref.exists() && rref.exists()) {
        return Err(format!(
            "missing extracted torch logits bins (run extract next to {}): {} / {}",
            path.display(),
            pref.display(),
            rref.display()
        ));
    }
    let parent_logits_torch = read_f32_bin(&pref, n_pos * PARENT_VOCAB)?;
    let ref_logits_torch = read_f32_bin(&rref, n_pos * PARENT_VOCAB)?;
    Ok(TorchStages {
        parent_logits_torch,
        ref_logits_torch,
    })
}

fn read_f32_bin(path: &Path, n: usize) -> Result<Vec<f32>, String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() != n * 4 {
        return Err(format!(
            "{}: got {} bytes want {}",
            path.display(),
            bytes.len(),
            n * 4
        ));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = f32::from_le_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ]);
    }
    Ok(out)
}

fn upload_f32(gpu: &mut Gpu, t: &GpuTensor, host: &[f32]) -> Result<(), String> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
    };
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .map_err(|e| format!("htod: {e:?}"))
}

fn metrics(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut diff2 = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
        let d = xf - yf;
        diff2 += d * d;
    }
    na = na.sqrt();
    nb = nb.sqrt();
    let cos = if na == 0.0 && nb == 0.0 {
        1.0
    } else {
        dot / (na * nb + 1e-300)
    };
    let rel = diff2.sqrt() / (nb + 1e-300);
    let nr = na / (nb + 1e-300);
    (cos, rel, nr)
}

fn argmax(a: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in a.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi
}

struct Args {
    model: PathBuf,
    parent_residual: PathBuf,
    torch_stages: Option<PathBuf>,
    ref_residual_bin: Option<PathBuf>,
    out: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut model = PathBuf::from(DEFAULT_MODEL);
    let mut parent_residual = PathBuf::from("/tmp/residual_content_parent/layer_42.f32");
    let mut torch_stages = Some(PathBuf::from("/tmp/head_path_content/head_path_stages.npz"));
    let mut ref_residual_bin = Some(PathBuf::from("/tmp/head_path_content/_ref_residual.f32"));
    let mut out = Some("/tmp/head_path_content/gpu_parent_head_compare.json".to_string());
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = PathBuf::from(it.next().ok_or("--model value")?),
            "--parent-residual" => {
                parent_residual = PathBuf::from(it.next().ok_or("--parent-residual value")?)
            }
            "--torch-stages" => {
                torch_stages = Some(PathBuf::from(it.next().ok_or("--torch-stages value")?))
            }
            "--ref-residual-bin" => {
                ref_residual_bin =
                    Some(PathBuf::from(it.next().ok_or("--ref-residual-bin value")?))
            }
            "--out" => out = Some(it.next().ok_or("--out value")?),
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4_parent_head_residual_compare --model DIR \\\n  --parent-residual layer_42.f32 --torch-stages stages.npz \\\n  --ref-residual-bin ref.f32 --out out.json"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    Ok(Args {
        model,
        parent_residual,
        torch_stages,
        ref_residual_bin,
        out,
    })
}
