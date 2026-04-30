//! MMQ vs WMMA bit-comparison diagnostic for Qwen3.5 GEMM call sites.
//!
//! Compares i8 WMMA + Q8_1 (MMQ) output against f16 WMMA output at each
//! GEMM call site in a transformer layer to diagnose which site/channel/layer
//! causes tool-call output corruption (ref: #87).
//!
//! Usage:
//!   cargo run --release --features deltanet --example channel_test_mmq -- \
//!       --model <path.hfq> [--stage site-scan|channel-map|layer-sweep]
//!       [--batch N] [--threshold F] [--layer N] [--site <name>]
//!
//! Stages:
//!   site-scan   (default) — sweep every (layer, site) pair; print error table
//!   channel-map — per-row error for a specific (site, layer) or all layers
//!   layer-sweep — per-layer error for a specific site across all layers
//!
//! MMQ is only available on RDNA3/3.5: gfx1100, gfx1101, gfx1102, gfx1103,
//! gfx1150, gfx1151. The binary exits 0 with a message on other archs.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("channel_test_mmq requires --features deltanet");
    std::process::exit(1);
}

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::qwen35;
    use std::path::Path;

    // ── CLI parsing ──────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "Usage: channel_test_mmq --model <path.hfq> \
             [--stage site-scan|channel-map|layer-sweep] \
             [--batch N] [--threshold F] [--layer N] [--site <name>]"
        );
        std::process::exit(0);
    }

    let mut model_path: Option<String> = None;
    let mut stage = "site-scan".to_string();
    let mut batch_size: usize = 128;
    let mut threshold: f32 = 0.01;
    let mut layer_filter: Option<usize> = None;
    let mut site_filter: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--stage" => {
                stage = args[i + 1].clone();
                i += 2;
            }
            "--batch" => {
                batch_size = args[i + 1].parse().expect("--batch must be integer");
                i += 2;
            }
            "--threshold" => {
                threshold = args[i + 1].parse().expect("--threshold must be float");
                i += 2;
            }
            "--layer" => {
                layer_filter = Some(args[i + 1].parse().expect("--layer must be integer"));
                i += 2;
            }
            "--site" => {
                site_filter = Some(args[i + 1].clone());
                i += 2;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }

    let model_path = model_path.expect("--model <path.hfq> is required");

    // ── GPU init ─────────────────────────────────────────────────────────
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");

    let mmq_archs = ["gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151"];
    if !mmq_archs.contains(&arch.as_str()) {
        eprintln!(
            "SKIP: MMQ requires RDNA3/3.5 (gfx1100..gfx1103, gfx1150, gfx1151). \
             Current arch: {arch}"
        );
        std::process::exit(0);
    }

    // ── Model loading ────────────────────────────────────────────────────
    eprintln!("Loading model: {model_path}");
    let hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config_from_hfq");
    eprintln!(
        "Config: dim={} layers={} heads={} kv_heads={} vocab={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size
    );
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load_weights");
    eprintln!("Weights loaded. Running stage: {stage}");

    // ── Dispatch stage ───────────────────────────────────────────────────
    match stage.as_str() {
        "site-scan" => {
            run_site_scan(&mut gpu, &weights, &config, batch_size, threshold);
        }
        "channel-map" => {
            let site = site_filter.expect("--site <name> required for channel-map");
            run_channel_map(&mut gpu, &weights, &config, batch_size, threshold, &site, layer_filter);
        }
        "layer-sweep" => {
            let site = site_filter.expect("--site <name> required for layer-sweep");
            run_layer_sweep(&mut gpu, &weights, &config, batch_size, threshold, &site);
        }
        other => {
            eprintln!("Unknown stage: {other}  (use site-scan | channel-map | layer-sweep)");
            std::process::exit(1);
        }
    }
}

// ── Data structures ──────────────────────────────────────────────────────────

/// Per-site aggregated error statistics.
#[cfg(feature = "deltanet")]
struct SiteStats {
    site: String,
    layer: usize,
    m: usize,
    k: usize,
    batch_size: usize,
    max_err: f32,
    mean_err: f32,
    bad_count: usize,
}

#[cfg(feature = "deltanet")]
impl SiteStats {
    fn header() {
        eprintln!(
            "{:<6}  {:<20}  {:>6}  {:>6}  {:>6}  {:>10}  {:>10}  {:>8}  {}",
            "layer", "site", "m", "k", "batch", "max_err", "mean_err", "bad", "status"
        );
        eprintln!("{}", "-".repeat(90));
    }

    fn print(&self) {
        let status = if self.bad_count > 0 { "FAIL" } else { "ok" };
        eprintln!(
            "{:<6}  {:<20}  {:>6}  {:>6}  {:>6}  {:>10.4e}  {:>10.4e}  {:>8}  {}",
            self.layer,
            self.site,
            self.m,
            self.k,
            self.batch_size,
            self.max_err,
            self.mean_err,
            self.bad_count,
            status,
        );
    }
}

/// Per-output-row error (for channel-map stage).
#[cfg(feature = "deltanet")]
struct RowStats {
    row: usize,
    max_err: f32,
    mean_err: f32,
    bad_count: usize,
}

// ── Helper: compute element-wise error stats ─────────────────────────────────

#[cfg(feature = "deltanet")]
fn compute_stats(
    y_ref: &[f32],
    y_mmq: &[f32],
    site: &str,
    layer: usize,
    m: usize,
    k: usize,
    batch_size: usize,
    threshold: f32,
) -> SiteStats {
    assert_eq!(y_ref.len(), y_mmq.len());
    let mut max_err = 0f32;
    let mut sum_err = 0f32;
    let mut bad_count = 0usize;
    for (a, b) in y_ref.iter().zip(y_mmq.iter()) {
        let e = (a - b).abs();
        if e > max_err {
            max_err = e;
        }
        sum_err += e;
        if e > threshold {
            bad_count += 1;
        }
    }
    let mean_err = if y_ref.is_empty() {
        0.0
    } else {
        sum_err / y_ref.len() as f32
    };
    SiteStats {
        site: site.to_string(),
        layer,
        m,
        k,
        batch_size,
        max_err,
        mean_err,
        bad_count,
    }
}

// ── Helper: seeded LCG synthetic activations ─────────────────────────────────

#[cfg(feature = "deltanet")]
fn synth_activations(batch_size: usize, k: usize, seed: u64) -> Vec<f32> {
    let n = batch_size * k;
    let mut out = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        // Simple LCG: Knuth constants
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to [-2.0, 2.0]
        let t = (state >> 33) as f32 / (u32::MAX as f32);
        out.push(t * 4.0 - 2.0);
    }
    out
}

// ── Helper: map site name → WeightTensor for a given layer ───────────────────

#[cfg(feature = "deltanet")]
fn get_weight_for_site<'a>(
    layer: &'a engine::qwen35::LayerWeights,
    site_name: &str,
) -> Option<&'a engine::llama::WeightTensor> {
    use engine::qwen35::LayerWeights;
    match layer {
        LayerWeights::DeltaNet(l) => match site_name {
            "qkvza.qkv"   => Some(&l.wqkv),
            "qkvza.z"     => Some(&l.wz),
            "qkvza.beta"  => Some(&l.w_beta),
            "qkvza.alpha" => Some(&l.w_alpha),
            "gate_up.gate" => Some(&l.w_gate),
            "gate_up.up"   => Some(&l.w_up),
            "residual"    => Some(&l.wo),
            _ => None,
        },
        LayerWeights::FullAttn(l) => match site_name {
            "qkv.q"       => Some(&l.wq),
            "qkv.k"       => Some(&l.wk),
            "qkv.v"       => Some(&l.wv),
            "gate_up.gate" => Some(&l.w_gate),
            "gate_up.up"   => Some(&l.w_up),
            "residual"    => Some(&l.wo),
            _ => None,
        },
        // MoE variants: same attention sites, no FFN sites (MoE uses routed experts)
        LayerWeights::DeltaNetMoe(l) => match site_name {
            "qkvza.qkv"   => Some(&l.wqkv),
            "qkvza.z"     => Some(&l.wz),
            "qkvza.beta"  => Some(&l.w_beta),
            "qkvza.alpha" => Some(&l.w_alpha),
            "residual"    => Some(&l.wo),
            _ => None,
        },
        LayerWeights::FullAttnMoe(l) => match site_name {
            "qkv.q"    => Some(&l.wq),
            "qkv.k"    => Some(&l.wk),
            "qkv.v"    => Some(&l.wv),
            "residual" => Some(&l.wo),
            _ => None,
        },
    }
}

/// Return the list of site names applicable to a given layer variant.
#[cfg(feature = "deltanet")]
fn sites_for_layer(layer: &engine::qwen35::LayerWeights) -> &'static [&'static str] {
    use engine::qwen35::LayerWeights;
    match layer {
        LayerWeights::DeltaNet(_) => &[
            "qkvza.qkv",
            "qkvza.z",
            "qkvza.beta",
            "qkvza.alpha",
            "gate_up.gate",
            "gate_up.up",
            "residual",
        ],
        LayerWeights::FullAttn(_) => &[
            "qkv.q",
            "qkv.k",
            "qkv.v",
            "gate_up.gate",
            "gate_up.up",
            "residual",
        ],
        LayerWeights::DeltaNetMoe(_) => &[
            "qkvza.qkv",
            "qkvza.z",
            "qkvza.beta",
            "qkvza.alpha",
            "residual",
        ],
        LayerWeights::FullAttnMoe(_) => &["qkv.q", "qkv.k", "qkv.v", "residual"],
    }
}

// ── Core comparison: WMMA vs MMQ for one weight ──────────────────────────────

/// Run WMMA then MMQ on identical inputs; return aggregated stats.
///
/// Sets `capture_mode = true` around the GEMM calls to force the WMMA/MMQ
/// dispatch paths (suppresses the rocBLAS fast path).
#[cfg(feature = "deltanet")]
fn compare_residual(
    gpu: &mut rdna_compute::Gpu,
    weight: &engine::llama::WeightTensor,
    x_data: &[f32],
    m: usize,
    k: usize,
    batch_size: usize,
    site_name: &str,
    layer: usize,
    threshold: f32,
) -> Result<SiteStats, String> {
    let (y_ref, y_mmq) = compare_residual_raw(gpu, weight, x_data, m, k, batch_size)?;
    Ok(compute_stats(
        &y_ref, &y_mmq, site_name, layer, m, k, batch_size, threshold,
    ))
}

/// Run WMMA then MMQ on identical inputs; return raw output vectors.
#[cfg(feature = "deltanet")]
fn compare_residual_raw(
    gpu: &mut rdna_compute::Gpu,
    weight: &engine::llama::WeightTensor,
    x_data: &[f32],
    m: usize,
    k: usize,
    batch_size: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    use rdna_compute::DType;
    use std::ffi::c_void;

    // Upload activations
    let x = gpu
        .upload_f32(x_data, &[batch_size, k])
        .map_err(|e| format!("upload x: {e}"))?;

    // Allocate output tensors (zeroed)
    let y_wmma = gpu
        .zeros(&[batch_size * m], DType::F32)
        .map_err(|e| format!("alloc y_wmma: {e}"))?;
    let y_mmq_buf = gpu
        .zeros(&[batch_size * m], DType::F32)
        .map_err(|e| format!("alloc y_mmq: {e}"))?;

    // Force WMMA/MMQ paths (skip rocBLAS fast path)
    gpu.capture_mode = true;

    // ── WMMA reference ───────────────────────────────────────────────────
    let r_wmma = gpu.gemm_hfq4g256_residual_wmma(&weight.buf, &x, &y_wmma, m, k, batch_size);

    // ── MMQ path ─────────────────────────────────────────────────────────
    let r_mmq = if r_wmma.is_ok() {
        let xq: *mut c_void = gpu
            .ensure_q8_1_mmq_x(&x, batch_size, k)
            .map_err(|e| {
                gpu.capture_mode = false;
                format!("ensure_q8_1_mmq_x: {e}")
            })?;
        gpu.gemm_hfq4g256_mmq_set_prequant(&weight.buf, xq, &y_mmq_buf, m, k, batch_size)
    } else {
        Ok(())
    };

    gpu.capture_mode = false;

    r_wmma.map_err(|e| format!("gemm_hfq4g256_residual_wmma: {e}"))?;
    r_mmq.map_err(|e| format!("gemm_hfq4g256_mmq_set_prequant: {e}"))?;

    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("device_synchronize: {e}"))?;

    let out_wmma = gpu
        .download_f32(&y_wmma)
        .map_err(|e| format!("download y_wmma: {e}"))?;
    let out_mmq = gpu
        .download_f32(&y_mmq_buf)
        .map_err(|e| format!("download y_mmq: {e}"))?;

    gpu.free_tensor(x).ok();
    gpu.free_tensor(y_wmma).ok();
    gpu.free_tensor(y_mmq_buf).ok();

    Ok((out_wmma, out_mmq))
}

// ── Per-row analysis ─────────────────────────────────────────────────────────

/// Compute per-output-row (weight row = output channel) error, sorted descending.
#[cfg(feature = "deltanet")]
fn per_row_diff(y_ref: &[f32], y_mmq: &[f32], m: usize, batch_size: usize) -> Vec<RowStats> {
    assert_eq!(y_ref.len(), batch_size * m);
    assert_eq!(y_mmq.len(), batch_size * m);

    let mut rows: Vec<RowStats> = (0..m)
        .map(|row| {
            let mut max_err = 0f32;
            let mut sum_err = 0f32;
            let mut bad_count = 0usize;
            for b in 0..batch_size {
                // Layout: y[batch, row] = data[batch * m + row]
                let idx = b * m + row;
                let e = (y_ref[idx] - y_mmq[idx]).abs();
                if e > max_err {
                    max_err = e;
                }
                sum_err += e;
                // threshold = 0.01 hard-coded for row-level diagnostics
                if e > 0.01 {
                    bad_count += 1;
                }
            }
            RowStats {
                row,
                max_err,
                mean_err: sum_err / batch_size as f32,
                bad_count,
            }
        })
        .collect();

    rows.sort_by(|a, b| b.max_err.partial_cmp(&a.max_err).unwrap_or(std::cmp::Ordering::Equal));
    rows
}

// ── Seed selection ───────────────────────────────────────────────────────────

/// Choose synth seed: residual (`wo`) has a different k than attention sites,
/// so we use a different seed to keep inputs independent across site types.
#[cfg(feature = "deltanet")]
fn seed_for_site(site_name: &str) -> u64 {
    if site_name == "residual" {
        0x1234_5678_9ABC_DEF0
    } else {
        0xDEAD_BEEF_CAFE_BABE
    }
}

// ── Stage: site-scan ─────────────────────────────────────────────────────────

#[cfg(feature = "deltanet")]
fn run_site_scan(
    gpu: &mut rdna_compute::Gpu,
    weights: &engine::qwen35::Qwen35Weights,
    _config: &engine::qwen35::Qwen35Config,
    batch_size: usize,
    threshold: f32,
) {
    eprintln!("\n=== site-scan: MMQ vs WMMA across all layers and sites ===");
    eprintln!("batch={batch_size}  threshold={threshold:.3e}");
    SiteStats::header();

    let mut all_stats: Vec<SiteStats> = Vec::new();
    let mut n_fail = 0usize;

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let sites = sites_for_layer(layer);
        for &site_name in sites {
            let weight = match get_weight_for_site(layer, site_name) {
                Some(w) => w,
                None => continue,
            };
            let m = weight.m;
            let k = weight.k;
            let seed = seed_for_site(site_name);
            let x_data = synth_activations(batch_size, k, seed);

            match compare_residual(gpu, weight, &x_data, m, k, batch_size, site_name, layer_idx, threshold) {
                Ok(stats) => {
                    stats.print();
                    if stats.bad_count > 0 {
                        n_fail += 1;
                    }
                    all_stats.push(stats);
                }
                Err(e) => {
                    eprintln!(
                        "  layer={layer_idx} site={site_name}: ERROR — {e}"
                    );
                }
            }
        }
    }

    // ── Top-10 worst sites ───────────────────────────────────────────────
    all_stats.sort_by(|a, b| b.max_err.partial_cmp(&a.max_err).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("\n--- Top-10 worst sites by max_err ---");
    SiteStats::header();
    for s in all_stats.iter().take(10) {
        s.print();
    }

    eprintln!("\n--- Summary ---");
    eprintln!("  Total sites scanned: {}", all_stats.len());
    eprintln!("  Sites exceeding threshold ({threshold:.3e}): {n_fail}");

    if n_fail > 0 {
        eprintln!("RESULT: FAIL — {n_fail} site(s) exceed threshold");
        std::process::exit(1);
    } else {
        eprintln!("RESULT: OK — all sites within threshold");
    }
}

// ── Stage: channel-map ───────────────────────────────────────────────────────

#[cfg(feature = "deltanet")]
fn run_channel_map(
    gpu: &mut rdna_compute::Gpu,
    weights: &engine::qwen35::Qwen35Weights,
    _config: &engine::qwen35::Qwen35Config,
    batch_size: usize,
    threshold: f32,
    site_name: &str,
    layer_filter: Option<usize>,
) {
    eprintln!("\n=== channel-map: per-row error for site={site_name} ===");
    eprintln!("batch={batch_size}  threshold={threshold:.3e}");

    let layers_to_scan: Vec<usize> = match layer_filter {
        Some(l) => vec![l],
        None => (0..weights.layers.len()).collect(),
    };

    for layer_idx in layers_to_scan {
        let layer = &weights.layers[layer_idx];
        let weight = match get_weight_for_site(layer, site_name) {
            Some(w) => w,
            None => {
                eprintln!("  layer={layer_idx}: site '{site_name}' not applicable (skipped)");
                continue;
            }
        };

        let m = weight.m;
        let k = weight.k;
        let seed = seed_for_site(site_name);
        let x_data = synth_activations(batch_size, k, seed);

        eprintln!("\n  layer={layer_idx}  m={m}  k={k}  batch={batch_size}");

        match compare_residual_raw(gpu, weight, &x_data, m, k, batch_size) {
            Err(e) => {
                eprintln!("    ERROR: {e}");
                continue;
            }
            Ok((y_ref, y_mmq)) => {
                let row_stats = per_row_diff(&y_ref, &y_mmq, m, batch_size);

                // Count rows exceeding threshold
                let bad_rows = row_stats.iter().filter(|r| r.max_err > threshold).count();
                eprintln!("    rows exceeding threshold ({threshold:.3e}): {bad_rows}/{m}");

                // Print top-20 worst rows
                eprintln!(
                    "    {:<6}  {:>10}  {:>10}  {:>8}",
                    "row", "max_err", "mean_err", "bad_acts"
                );
                eprintln!("    {}", "-".repeat(42));
                for rs in row_stats.iter().take(20) {
                    eprintln!(
                        "    {:<6}  {:>10.4e}  {:>10.4e}  {:>8}",
                        rs.row, rs.max_err, rs.mean_err, rs.bad_count
                    );
                }
            }
        }
    }
}

// ── Stage: layer-sweep ───────────────────────────────────────────────────────

#[cfg(feature = "deltanet")]
fn run_layer_sweep(
    gpu: &mut rdna_compute::Gpu,
    weights: &engine::qwen35::Qwen35Weights,
    _config: &engine::qwen35::Qwen35Config,
    batch_size: usize,
    threshold: f32,
    site_name: &str,
) {
    eprintln!("\n=== layer-sweep: per-layer error for site={site_name} ===");
    eprintln!("batch={batch_size}  threshold={threshold:.3e}");

    SiteStats::header();
    let mut all_stats: Vec<SiteStats> = Vec::new();

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let weight = match get_weight_for_site(layer, site_name) {
            Some(w) => w,
            None => continue, // site not applicable to this layer type
        };

        let m = weight.m;
        let k = weight.k;
        let seed = seed_for_site(site_name);
        let x_data = synth_activations(batch_size, k, seed);

        match compare_residual(gpu, weight, &x_data, m, k, batch_size, site_name, layer_idx, threshold) {
            Ok(stats) => {
                stats.print();
                all_stats.push(stats);
            }
            Err(e) => {
                eprintln!("  layer={layer_idx}: ERROR — {e}");
            }
        }
    }

    if all_stats.is_empty() {
        eprintln!("\nNo layers found with site '{site_name}'");
        return;
    }

    // Identify worst layer
    let worst = all_stats
        .iter()
        .max_by(|a, b| a.max_err.partial_cmp(&b.max_err).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    eprintln!("\n--- Summary ---");
    eprintln!(
        "  Layers scanned: {}  (with site='{site_name}')",
        all_stats.len()
    );
    eprintln!(
        "  Worst layer: {}  max_err={:.4e}  bad_count={}",
        worst.layer, worst.max_err, worst.bad_count
    );
    let n_fail = all_stats.iter().filter(|s| s.bad_count > 0).count();
    eprintln!("  Layers exceeding threshold: {n_fail}");
}
