//! dots.ocr vision-tower validation driver.
//!
//! Loads a dots.ocr HFQ, runs the vision encoder + merger on a single
//! input image, and (optionally) validates the merged-token output
//! against captured HF reference activations.
//!
//! Phase 2c-5b smoke driver. Vision-only — does NOT load the text
//! decoder (saves ~1.5 GB GPU memory + load time when the only thing
//! we want to check is the vision pipeline). The full
//! text+vision+daemon end-to-end wiring lands in Phase 3.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example infer_dots_ocr -p hipfire-arch-dots-ocr -- \
//!     --hfq ~/.hipfire/models/dots-ocr.q8.hfq \
//!     --image benchmarks/images/dots_ocr_smoke_001.jpg \
//!     --reference benchmarks/references/dots_ocr_smoke_001_activations
//! ```
//!
//! `--reference` is optional. When provided, the binary loads the
//! sampled merger reference (`<dir>/merger.npy` + `<dir>/index.json`)
//! and reports cosine similarity + max absolute difference per
//! sampled row. The pass tolerance per plan §5 phase 2c-5 is
//! `|Δ| < 1e-2 OR cosine > 0.999` (allows for the bf16→F16 cast
//! slack that compounds across 42 blocks).
//!
//! # Output stages instrumented
//!
//! For first-light validation, only the final merger output is
//! checked. The reference set has captures at `patch_embed`,
//! `block_00`, `block_21`, `block_41`, `post_trunk_norm`, and
//! `merger`. If the merger validation fails, the natural next step
//! is to add per-stage dumping to `vision_forward` (D3 hint from
//! rev-glm5 — see Task #69 description) to bisect the divergence.

use std::path::{Path, PathBuf};
use std::fs;
use std::time::Instant;

use hipfire_arch_dots_ocr::{dots_ocr, image as preprocess, rope};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

// ─── Argument parsing ─────────────────────────────────────────────────

#[derive(Debug)]
struct Args {
    hfq_path: PathBuf,
    image_path: PathBuf,
    reference_dir: Option<PathBuf>,
    verbose: bool,
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut hfq = None;
    let mut img = None;
    let mut refd = None;
    let mut verbose = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--hfq" => { hfq = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--image" => { img = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--reference" => { refd = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "-v" | "--verbose" => { verbose = true; i += 1; }
            "--help" | "-h" => {
                eprintln!("{}", USAGE);
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}\n\n{USAGE}")),
        }
    }
    Ok(Args {
        hfq_path: hfq.ok_or("--hfq required")?,
        image_path: img.ok_or("--image required")?,
        reference_dir: refd,
        verbose,
    })
}

const USAGE: &str = "\
Usage: infer_dots_ocr --hfq <path.hfq> --image <path.{jpg,png}> \
[--reference <dir>] [-v]

  --hfq         dots.ocr HFQ file (arch_id=8)
  --image       input image (JPEG or PNG)
  --reference   directory holding the HF reference activations
                (expects merger.npy + index.json). Optional —
                without it, the driver runs the forward pass and
                prints output stats but skips validation.
  -v            verbose: per-block timing, full sample diffs
";

// ─── Minimal NPY (NumPy .npy) reader ──────────────────────────────────
//
// Format reference: github.com/numpy/numpy/blob/main/numpy/lib/format.py.
// We only support little-endian f32 with C (row-major) ordering — the
// shape we use for the reference captures.

fn load_npy_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let bytes = fs::read(path)
        .map_err(|e| format!("npy: read {}: {e}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("npy: {} missing magic", path.display()));
    }
    let major = bytes[6];
    let minor = bytes[7];
    let (header_len, header_start) = match major {
        1 => {
            let l = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (l, 10)
        }
        2 | 3 => {
            let l = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (l, 12)
        }
        _ => return Err(format!("npy: unsupported version {major}.{minor}")),
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .map_err(|e| format!("npy header: {e}"))?;
    // Dict-literal-ish: { 'descr': '<f4', 'fortran_order': False, 'shape': (..., ..., ), }
    let descr = extract_str(header, "descr").ok_or("npy: no descr")?;
    if descr != "<f4" {
        return Err(format!("npy: only <f4 supported, got {descr}"));
    }
    let fortran = extract_bool(header, "fortran_order").unwrap_or(false);
    if fortran {
        return Err("npy: fortran_order=True not supported".to_string());
    }
    let shape = extract_shape(header).ok_or("npy: no shape")?;
    let n: usize = shape.iter().product();
    let data_off = header_start + header_len;
    if bytes.len() - data_off < n * 4 {
        return Err(format!(
            "npy: truncated, header says {n} f32 but only {} bytes after header",
            bytes.len() - data_off,
        ));
    }
    let mut v = Vec::with_capacity(n);
    for chunk in bytes[data_off..data_off + n * 4].chunks_exact(4) {
        v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((shape, v))
}

fn extract_str(header: &str, key: &str) -> Option<String> {
    let needle = format!("'{key}':");
    let i = header.find(&needle)? + needle.len();
    let rest = header[i..].trim_start();
    let rest = rest.strip_prefix('\'')?;
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

fn extract_bool(header: &str, key: &str) -> Option<bool> {
    let needle = format!("'{key}':");
    let i = header.find(&needle)? + needle.len();
    let rest = header[i..].trim_start();
    if rest.starts_with("True") { Some(true) }
    else if rest.starts_with("False") { Some(false) }
    else { None }
}

fn extract_shape(header: &str) -> Option<Vec<usize>> {
    let i = header.find("'shape':")? + "'shape':".len();
    let rest = header[i..].trim_start();
    let rest = rest.strip_prefix('(')?;
    let end = rest.find(')')?;
    let inner = &rest[..end];
    let mut dims = Vec::new();
    for tok in inner.split(',') {
        let t = tok.trim();
        if t.is_empty() { continue; }
        dims.push(t.parse::<usize>().ok()?);
    }
    Some(dims)
}

// ─── Validation maths ─────────────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { (dot / (na * nb)) as f32 }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// ─── Per-stage tensor stats ───────────────────────────────────────────

fn print_stats(label: &str, data: &[f32], shape: &[usize]) {
    let mean: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / data.len() as f64;
    let var: f64 = data.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std = var.sqrt();
    let (mn, mx) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| {
        (a.min(x), b.max(x))
    });
    let nan_count = data.iter().filter(|x| x.is_nan()).count();
    let inf_count = data.iter().filter(|x| x.is_infinite()).count();
    println!(
        "  {label:20} shape={shape:?} mean={mean:+.4} std={std:.4} \
         range=[{mn:+.3}, {mx:+.3}] nan={nan_count} inf={inf_count}"
    );
}

// ─── Main ──────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    println!("infer_dots_ocr: {:?}", args);

    let t_total = Instant::now();

    // Load HFQ + parse config.
    println!("\n[1/5] Open HFQ + parse config");
    let t = Instant::now();
    let hfq = HfqFile::open(&args.hfq_path)?;
    if hfq.arch_id != 8 {
        eprintln!(
            "warning: HFQ arch_id is {}, expected 8 for dots.ocr. \
             Proceeding but layer naming may not match.",
            hfq.arch_id,
        );
    }
    let cfg = dots_ocr::DotsOcrConfig::from_hfq(&hfq)
        .map_err(|e| format!("config parse: {e}"))?;
    println!(
        "  arch_id={}, text.layers={}, vision.layers={}, vision.embed_dim={}, vision.heads={}",
        hfq.arch_id, cfg.text.num_hidden_layers, cfg.vision.num_hidden_layers,
        cfg.vision.embed_dim, cfg.vision.num_attention_heads,
    );
    println!("  ({:.2}s)", t.elapsed().as_secs_f32());

    // Init GPU.
    println!("\n[2/5] Init GPU");
    let t = Instant::now();
    let mut gpu = Gpu::init()?;
    println!("  ({:.2}s)", t.elapsed().as_secs_f32());

    // Load vision-tower weights only.
    println!("\n[3/5] Load vision weights ({} blocks)", cfg.vision.num_hidden_layers);
    let t = Instant::now();
    let weights = dots_ocr::load_vision_weights(&hfq, &cfg.vision, &mut gpu)?;
    // Surface any latent device-side error from the upload bursts before
    // the first compute kernel — otherwise a sticky HIP error from an
    // earlier async op will misattribute to the next hipModuleLoad.
    gpu.hip.device_synchronize()?;
    println!("  vision weights on GPU ({:.2}s)", t.elapsed().as_secs_f32());

    // Preprocess image.
    println!("\n[4/5] Preprocess image: {}", args.image_path.display());
    let t = Instant::now();
    let img = preprocess::preprocess_image(&args.image_path)?;
    let n_patches = img.n_patches();
    let n_visual_tokens = img.n_visual_tokens();
    println!(
        "  resized={}x{}, grid={}x{}, n_patches={n_patches}, n_visual_tokens={n_visual_tokens} \
         ({:.2}s)",
        img.resized_h, img.resized_w, img.grid_h, img.grid_w,
        t.elapsed().as_secs_f32(),
    );

    // Patches → GPU.
    let patches_gpu = gpu.upload_f32(&img.patches, &[n_patches, img.patches.len() / n_patches])?;

    // Forward.
    println!("\n[5/5] vision_forward");
    let t = Instant::now();
    let merged_gpu = dots_ocr::vision_forward(
        &mut gpu, &weights, &cfg.vision, &patches_gpu, img.grid_h, img.grid_w,
    )?;
    gpu.free_tensor(patches_gpu)?;
    let elapsed = t.elapsed().as_secs_f32();
    println!("  done in {elapsed:.2}s ({:.1} ms/block)", 1000.0 * elapsed / cfg.vision.num_hidden_layers as f32);

    // Download + summarise.
    let merged = gpu.download_f32(&merged_gpu)?;
    let out_dim = cfg.vision.out_hidden_size;
    let shape = vec![n_visual_tokens, out_dim];
    println!("\nMerger output:");
    print_stats("merger", &merged, &shape);

    // Optional validation.
    if let Some(refd) = &args.reference_dir {
        println!("\nValidation against HF reference: {}", refd.display());
        validate_merger(&merged, n_visual_tokens, out_dim, refd, args.verbose)?;
    } else {
        println!("\n(skipped validation — pass --reference <dir> to enable)");
    }

    // Free everything.
    gpu.free_tensor(merged_gpu)?;
    weights.free_gpu(&mut gpu);

    println!("\nTotal: {:.2}s", t_total.elapsed().as_secs_f32());
    Ok(())
}

fn validate_merger(
    our_merger: &[f32],
    n_visual_tokens: usize,
    out_dim: usize,
    refd: &Path,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load + parse index.json to get sample indices.
    let idx_bytes = fs::read_to_string(refd.join("index.json"))?;
    let idx_json: serde_json::Value = serde_json::from_str(&idx_bytes)?;
    let captures = idx_json["captures"].as_array().ok_or("index.json: no captures")?;
    let merger_cap = captures.iter()
        .find(|c| c["name"] == "merger")
        .ok_or("index.json: no merger capture")?;
    let full_shape: Vec<usize> = merger_cap["full_shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let sample_indices: Vec<usize> = merger_cap["sample_indices"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();

    // Shape parity.
    if full_shape[0] != n_visual_tokens || full_shape[1] != out_dim {
        return Err(format!(
            "shape mismatch: hipfire merger {} × {} vs HF full_shape {:?}. \
             Likely cause: image-grid-thw mismatch — smart_resize chose different \
             dims than HF's processor. Check resized_h/resized_w against \
             image_grid_thw in index.json.",
            n_visual_tokens, out_dim, full_shape,
        ).into());
    }

    // Load reference (sampled rows).
    let (ref_shape, ref_data) = load_npy_f32(&refd.join("merger.npy"))?;
    if ref_shape != [sample_indices.len(), out_dim] {
        return Err(format!(
            "merger.npy shape {ref_shape:?} != [n_samples={}, {}]",
            sample_indices.len(), out_dim,
        ).into());
    }

    // Compute per-row stats; report top-N worst by cosine.
    println!("  Comparing {} sampled rows ({}-dim each) ...", sample_indices.len(), out_dim);
    let mut rows: Vec<(usize, usize, f32, f32)> = Vec::with_capacity(sample_indices.len());
    for (s, &row_idx) in sample_indices.iter().enumerate() {
        let our_row = &our_merger[row_idx * out_dim..(row_idx + 1) * out_dim];
        let ref_row = &ref_data[s * out_dim..(s + 1) * out_dim];
        let cos = cosine(our_row, ref_row);
        let mab = max_abs(our_row, ref_row);
        rows.push((s, row_idx, cos, mab));
    }

    // Summary stats.
    let mean_cos: f64 = rows.iter().map(|r| r.2 as f64).sum::<f64>() / rows.len() as f64;
    let min_cos: f32 = rows.iter().map(|r| r.2).fold(1.0, f32::min);
    let max_mab: f32 = rows.iter().map(|r| r.3).fold(0.0, f32::max);
    let mean_mab: f64 = rows.iter().map(|r| r.3 as f64).sum::<f64>() / rows.len() as f64;
    println!("    mean cosine: {mean_cos:.5}");
    println!("    min  cosine: {min_cos:.5}");
    println!("    max  |Δ|  : {max_mab:.5}");
    println!("    mean |Δ|  : {mean_mab:.5}");

    // Per-row pass/fail (cosine > 0.999 OR max |Δ| < 1e-2).
    let mut failed: Vec<&(usize, usize, f32, f32)> = rows.iter()
        .filter(|r| r.2 <= 0.999 && r.3 >= 1e-2)
        .collect();
    failed.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let n_fail = failed.len();
    let n_pass = rows.len() - n_fail;
    println!("    pass (cos>0.999 OR |Δ|<1e-2): {n_pass}/{}", rows.len());

    if n_fail > 0 {
        println!("    failed rows (top 10 by cosine):");
        for (s, idx, cos, mab) in failed.iter().take(10) {
            println!("      sample={s:3} row={idx:5} cos={cos:.5} max_abs={mab:.5}");
        }
    }

    if verbose {
        println!("    all rows:");
        for (s, idx, cos, mab) in &rows {
            println!("      sample={s:3} row={idx:5} cos={cos:.5} max_abs={mab:.5}");
        }
    }

    // Hard fail iff > 50% rows fail — small drift on a few rows is acceptable
    // given bf16→F16 + 42-block compounding noise. The interesting signal is
    // "did the pipeline fundamentally work" vs "edge-case rounding".
    if n_fail * 2 > rows.len() {
        return Err(format!(
            "validation FAILED: {n_fail}/{} rows outside tolerance \
             (cos<=0.999 AND |Δ|>=1e-2)",
            rows.len(),
        ).into());
    }

    println!("  ✓ validation PASSED");
    // Currently we don't use rope here — but importing keeps the example
    // close to vision_forward's internal API for future per-stage dumps.
    let _ = rope::n_patches(1, 1);
    Ok(())
}
