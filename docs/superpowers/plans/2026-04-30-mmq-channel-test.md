# MMQ Channel-Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a diagnostic binary that compares MMQ (i8 WMMA) vs f16 WMMA output at each GEMM call site to identify the source of tool-call corruption (Kaden-Schutt/hipfire#87).

**Architecture:** Single example binary (`channel_test_mmq.rs`) loads a real model, creates synthetic activations matching prefill batch size, and replays each layer's weight matrices through both the f16 WMMA and MMQ GEMM paths. Compares outputs on CPU with three progressive stages: site-scan (which site?), channel-map (which rows?), layer-sweep (which layers?).

**Tech Stack:** Rust, existing `rdna-compute`/`engine` crate APIs. No new dependencies. No changes to `dispatch.rs`.

---

### Task 1: Scaffold the binary with arg parsing and model loading

**Files:**
- Create: `crates/engine/examples/channel_test_mmq.rs`

- [ ] **Step 1: Create the binary with arg parsing and model load**

```rust
//! MMQ vs f16-WMMA bit-comparison analysis for #87 auto-MMQ regression.
//!
//! Three stages:
//!   site-scan    — which of the 4 GEMM sites is worst?
//!   channel-map  — which output rows (channels) in a site are worst?
//!   layer-sweep  — which layers concentrate the error?
//!
//! Usage:
//!   cargo run --release --features deltanet --example channel_test_mmq -- \
//!     --model models/qwen3.6-27b.mq4 --stage site-scan [--batch 128] [--threshold 0.01]
//!     --stage channel-map [--layer 14] [--site residual]
//!     --stage layer-sweep [--site residual]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("Build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::llama::{self, WeightTensor, DType};
    use engine::qwen35::{self, LayerWeights, Qwen35Config};
    use rdna_compute::{Gpu, GpuTensor};
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    let model_path = args.iter().position(|a| a == "--model")
        .map(|i| args[i + 1].as_str())
        .unwrap_or_else(|| { eprintln!("Usage: --model <path> --stage <site-scan|channel-map|layer-sweep>"); std::process::exit(1) });
    let stage = args.iter().position(|a| a == "--stage")
        .map(|i| args[i + 1].as_str())
        .unwrap_or("site-scan");
    let batch_size: usize = args.iter().position(|a| a == "--batch")
        .and_then(|i| args[i + 1].parse().ok())
        .unwrap_or(128);
    let threshold: f32 = args.iter().position(|a| a == "--threshold")
        .and_then(|i| args[i + 1].parse().ok())
        .unwrap_or(0.01);
    let filter_layer: Option<usize> = args.iter().position(|a| a == "--layer")
        .and_then(|i| args[i + 1].parse().ok());
    let filter_site: Option<&str> = args.iter().position(|a| a == "--site")
        .map(|i| args[i + 1].as_str());

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);

    // MMQ only compiles on RDNA3/3.5
    let arch = gpu.arch.clone();
    if !matches!(arch.as_str(), "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1150" | "gfx1151") {
        eprintln!("SKIP: MMQ i8 WMMA requires RDNA3/3.5 (got {})", arch);
        std::process::exit(0);
    }

    eprintln!("Loading {}...", model_path);
    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("weights");
    eprintln!("Loaded: {} layers, dim={}, hidden_dim={}", config.n_layers, config.dim, config.hidden_dim);

    match stage {
        "site-scan" => site_scan(&mut gpu, &weights, &config, batch_size, threshold),
        "channel-map" => channel_map(&mut gpu, &weights, &config, batch_size, threshold, filter_layer, filter_site),
        "layer-sweep" => layer_sweep(&mut gpu, &weights, &config, batch_size, threshold, filter_site),
        _ => { eprintln!("Unknown stage: {stage}. Use site-scan|channel-map|layer-sweep"); std::process::exit(1); }
    }
}

#[cfg(feature = "deltanet")]
fn site_scan(_gpu: &mut rdna_compute::Gpu, _w: &engine::qwen35::Qwen35Weights, _c: &engine::qwen35::Qwen35Config, _batch: usize, _thresh: f32) {
    eprintln!("site-scan: TODO");
}

#[cfg(feature = "deltanet")]
fn channel_map(_gpu: &mut rdna_compute::Gpu, _w: &engine::qwen35::Qwen35Weights, _c: &engine::qwen35::Qwen35Config, _batch: usize, _thresh: f32, _layer: Option<usize>, _site: Option<&str>) {
    eprintln!("channel-map: TODO");
}

#[cfg(feature = "deltanet")]
fn layer_sweep(_gpu: &mut rdna_compute::Gpu, _w: &engine::qwen35::Qwen35Weights, _c: &engine::qwen35::Qwen35Config, _batch: usize, _thresh: f32, _site: Option<&str>) {
    eprintln!("layer-sweep: TODO");
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build --release --features deltanet --example channel_test_mmq 2>&1 | tail -5`
Expected: Build succeeds (possibly with dead-code warnings for the TODO stubs).

- [ ] **Step 3: Verify it loads a model and prints config**

Run: `cargo run --release --features deltanet --example channel_test_mmq -- --model models/qwen3.5-9b.mq4 --stage site-scan 2>&1 | head -10`
Expected: Prints `GPU: gfx1151`, `Loaded: N layers, dim=...`, then `site-scan: TODO`.

If you don't have `qwen3.5-9b.mq4`, use whichever `.mq4` model is available in `models/`. The 9B is preferred for fast iteration; 27B is the canonical reproducer but loads slower.

- [ ] **Step 4: Commit**

```bash
git add crates/engine/examples/channel_test_mmq.rs
git commit -m "feat(diag): scaffold channel_test_mmq binary for #87 MMQ analysis"
```

---

### Task 2: Implement synthetic activation generator and comparison helpers

These are the shared building blocks all three stages use.

**Files:**
- Modify: `crates/engine/examples/channel_test_mmq.rs`

- [ ] **Step 1: Add the comparison stats struct and diff function**

Add these above the stage functions, inside the `#[cfg(feature = "deltanet")]` block (or better: put all the `#[cfg(feature = "deltanet")]` logic in a module block):

```rust
#[cfg(feature = "deltanet")]
struct SiteStats {
    site: String,
    layer: usize,
    m: usize,
    k: usize,
    batch_size: usize,
    max_err: f32,
    mean_err: f32,
    bad_count: usize,   // elements exceeding threshold
    threshold: f32,
}

#[cfg(feature = "deltanet")]
impl SiteStats {
    fn header() {
        eprintln!("{:>5} | {:>10} | {:>10} | {:>10} | {:>8} | {:>12}",
            "Layer", "Site", "MaxErr", "MeanErr", ">Thresh", "Shape");
        eprintln!("{}", "-".repeat(68));
    }

    fn print(&self) {
        let flag = if self.bad_count > 0 { " <<" } else { "" };
        eprintln!("{:>5} | {:>10} | {:>10.6} | {:>10.6} | {:>8} | {:>5}x{:<5}{}",
            self.layer, self.site, self.max_err, self.mean_err,
            self.bad_count, self.m, self.k, flag);
    }
}

#[cfg(feature = "deltanet")]
fn compute_stats(y_ref: &[f32], y_mmq: &[f32], site: &str, layer: usize, m: usize, k: usize, batch_size: usize, threshold: f32) -> SiteStats {
    let mut max_err: f32 = 0.0;
    let mut sum_err: f64 = 0.0;
    let mut bad_count: usize = 0;
    let n = y_ref.len().min(y_mmq.len());
    for i in 0..n {
        let err = (y_ref[i] - y_mmq[i]).abs();
        max_err = max_err.max(err);
        sum_err += err as f64;
        if err > threshold { bad_count += 1; }
    }
    let mean_err = if n > 0 { (sum_err / n as f64) as f32 } else { 0.0 };
    SiteStats { site: site.to_string(), layer, m, k, batch_size, max_err, mean_err, bad_count, threshold }
}
```

- [ ] **Step 2: Add the synthetic activation generator**

The activations must be realistic enough to trigger the Q8_1 precision issue.
We use pseudo-random values in a range typical of RMSNorm output (~[-2, 2]):

```rust
/// Generate synthetic activation data that mimics post-RMSNorm hidden states.
/// Uses a seeded LCG so results are reproducible across runs.
#[cfg(feature = "deltanet")]
fn synth_activations(batch_size: usize, k: usize, seed: u64) -> Vec<f32> {
    let n = batch_size * k;
    let mut out = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Map to [-2.0, 2.0] — typical RMSNorm output range
        let bits = ((state >> 33) & 0xFFFF_FFFF) as u32;
        let val = (bits as f32 / u32::MAX as f32) * 4.0 - 2.0;
        out.push(val);
    }
    out
}
```

- [ ] **Step 3: Add the per-site comparison runner**

This is the core function: takes a weight tensor, an activation vector, calls
both f16 WMMA and MMQ on the same input, downloads results, diffs.

```rust
/// Compare f16-WMMA vs MMQ output for a single weight matrix (residual-style GEMM).
/// Returns stats. `y` output is `[batch_size, m]` (row-major, batch-minor).
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
) -> SiteStats {
    use rdna_compute::DType;

    let x_gpu = gpu.upload_f32(x_data, &[batch_size * k]).unwrap();
    let y_wmma_data = vec![0.0f32; batch_size * m];
    let y_mmq_data = vec![0.0f32; batch_size * m];
    let y_wmma = gpu.upload_f32(&y_wmma_data, &[batch_size * m]).unwrap();
    let y_mmq = gpu.upload_f32(&y_mmq_data, &[batch_size * m]).unwrap();

    // f16 WMMA reference
    gpu.capture_mode = true; // skip rocBLAS
    gpu.gemm_hfq4g256_residual_wmma(&weight.buf, &x_gpu, &y_wmma, m, k, batch_size).unwrap();
    gpu.hip.device_synchronize().unwrap();

    // MMQ path
    let xq = gpu.ensure_q8_1_mmq_x(&x_gpu, batch_size, k).unwrap();
    gpu.gemm_hfq4g256_mmq_set_prequant(&weight.buf, xq, &y_mmq, m, k, batch_size).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.capture_mode = false;

    let ref_out = gpu.download_f32(&y_wmma).unwrap();
    let mmq_out = gpu.download_f32(&y_mmq).unwrap();

    let _ = gpu.free_tensor(x_gpu);
    let _ = gpu.free_tensor(y_wmma);
    let _ = gpu.free_tensor(y_mmq);

    compute_stats(&ref_out, &mmq_out, site_name, layer, m, k, batch_size, threshold)
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo build --release --features deltanet --example channel_test_mmq 2>&1 | tail -5`
Expected: Build succeeds. There may be warnings about `ensure_q8_1_mmq_x` and
`gemm_hfq4g256_mmq_set_prequant` visibility — if they are `pub` this works; if
they are `fn` (private), we need to check.

**If `ensure_q8_1_mmq_x` is private:** It is — it's declared as `fn` not `pub fn`
at line 906. We need to make it `pub` (and `gemm_hfq4g256_mmq_set_prequant` at
line 4861). This is a minimal, non-behavioral change to `dispatch.rs`:

```rust
// dispatch.rs line 906: change `fn` to `pub fn`
pub fn ensure_q8_1_mmq_x(...)

// dispatch.rs line 4861: change `fn` to `pub fn`
pub fn gemm_hfq4g256_mmq_set_prequant(...)
```

- [ ] **Step 5: Commit**

```bash
git add crates/engine/examples/channel_test_mmq.rs crates/rdna-compute/src/dispatch.rs
git commit -m "feat(diag): add comparison helpers and synth activation gen for channel_test_mmq"
```

---

### Task 3: Implement site-scan stage

**Files:**
- Modify: `crates/engine/examples/channel_test_mmq.rs`

- [ ] **Step 1: Implement `site_scan`**

Replace the TODO stub with:

```rust
#[cfg(feature = "deltanet")]
fn site_scan(
    gpu: &mut rdna_compute::Gpu,
    weights: &engine::qwen35::Qwen35Weights,
    config: &engine::qwen35::Qwen35Config,
    batch_size: usize,
    threshold: f32,
) {
    use engine::qwen35::LayerWeights;

    eprintln!("\n=== STAGE 1: site-scan (batch_size={batch_size}, threshold={threshold}) ===\n");

    let k = config.dim;
    let x_data = synth_activations(batch_size, k, 0xDEAD_BEEF_CAFE_BABEu64);

    let mut all_stats: Vec<SiteStats> = Vec::new();
    SiteStats::header();

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        match layer {
            LayerWeights::DeltaNet(l) => {
                // QKVZA — 4-way fused: test each sub-projection individually
                let s = compare_residual(gpu, &l.wqkv, &x_data, l.wqkv.m, k, batch_size, "qkvza.qkv", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.wz, &x_data, l.wz.m, k, batch_size, "qkvza.z", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.w_beta, &x_data, l.w_beta.m, k, batch_size, "qkvza.beta", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.w_alpha, &x_data, l.w_alpha.m, k, batch_size, "qkvza.alpha", layer_idx, threshold);
                s.print(); all_stats.push(s);
                // gate_up
                let s = compare_residual(gpu, &l.w_gate, &x_data, l.w_gate.m, k, batch_size, "gate_up.gate", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.w_up, &x_data, l.w_up.m, k, batch_size, "gate_up.up", layer_idx, threshold);
                s.print(); all_stats.push(s);
                // residual (wo)
                // wo has different k (d_inner, not dim) — use the weight's own k
                let s = compare_residual(gpu, &l.wo, &synth_activations(batch_size, l.wo.k, 0x1234_5678_9ABC_DEF0u64),
                    l.wo.m, l.wo.k, batch_size, "residual", layer_idx, threshold);
                s.print(); all_stats.push(s);
            }
            LayerWeights::FullAttn(l) => {
                // QKV — 3-way
                let s = compare_residual(gpu, &l.wq, &x_data, l.wq.m, k, batch_size, "qkv.q", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.wk, &x_data, l.wk.m, k, batch_size, "qkv.k", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.wv, &x_data, l.wv.m, k, batch_size, "qkv.v", layer_idx, threshold);
                s.print(); all_stats.push(s);
                // gate_up
                let s = compare_residual(gpu, &l.w_gate, &x_data, l.w_gate.m, k, batch_size, "gate_up.gate", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.w_up, &x_data, l.w_up.m, k, batch_size, "gate_up.up", layer_idx, threshold);
                s.print(); all_stats.push(s);
                // residual (wo)
                let s = compare_residual(gpu, &l.wo, &synth_activations(batch_size, l.wo.k, 0x1234_5678_9ABC_DEF0u64),
                    l.wo.m, l.wo.k, batch_size, "residual", layer_idx, threshold);
                s.print(); all_stats.push(s);
            }
            LayerWeights::DeltaNetMoe(l) => {
                // Same attention sites, skip MoE FFN
                let s = compare_residual(gpu, &l.wqkv, &x_data, l.wqkv.m, k, batch_size, "qkvza.qkv", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.wz, &x_data, l.wz.m, k, batch_size, "qkvza.z", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.w_beta, &x_data, l.w_beta.m, k, batch_size, "qkvza.beta", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.w_alpha, &x_data, l.w_alpha.m, k, batch_size, "qkvza.alpha", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.wo, &synth_activations(batch_size, l.wo.k, 0x1234_5678_9ABC_DEF0u64),
                    l.wo.m, l.wo.k, batch_size, "residual", layer_idx, threshold);
                s.print(); all_stats.push(s);
                eprintln!("{:>5} | {:>10} | (MoE FFN skipped)", layer_idx, "gate_up");
            }
            LayerWeights::FullAttnMoe(l) => {
                let s = compare_residual(gpu, &l.wq, &x_data, l.wq.m, k, batch_size, "qkv.q", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.wk, &x_data, l.wk.m, k, batch_size, "qkv.k", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.wv, &x_data, l.wv.m, k, batch_size, "qkv.v", layer_idx, threshold);
                s.print(); all_stats.push(s);
                let s = compare_residual(gpu, &l.wo, &synth_activations(batch_size, l.wo.k, 0x1234_5678_9ABC_DEF0u64),
                    l.wo.m, l.wo.k, batch_size, "residual", layer_idx, threshold);
                s.print(); all_stats.push(s);
                eprintln!("{:>5} | {:>10} | (MoE FFN skipped)", layer_idx, "gate_up");
            }
        }
    }

    // Summary: top-10 worst
    all_stats.sort_by(|a, b| b.max_err.partial_cmp(&a.max_err).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("\n=== TOP 10 WORST (site, layer) ===\n");
    SiteStats::header();
    for s in all_stats.iter().take(10) {
        s.print();
    }

    let any_bad = all_stats.iter().any(|s| s.bad_count > 0);
    if any_bad {
        eprintln!("\nFAIL: {} (site, layer) pairs exceed threshold {}",
            all_stats.iter().filter(|s| s.bad_count > 0).count(), threshold);
        std::process::exit(1);
    } else {
        eprintln!("\nOK: no elements exceed threshold {threshold}");
    }
}
```

- [ ] **Step 2: Verify it compiles and runs**

Run: `cargo build --release --features deltanet --example channel_test_mmq 2>&1 | tail -5`
Expected: Compiles.

Run: `cargo run --release --features deltanet --example channel_test_mmq -- --model models/qwen3.5-9b.mq4 --stage site-scan 2>&1 | tail -30`
Expected: Prints the comparison table for all layers × sites. On gfx1151, some sites
should show non-zero error. Record which sites are worst.

- [ ] **Step 3: Commit**

```bash
git add crates/engine/examples/channel_test_mmq.rs
git commit -m "feat(diag): implement site-scan stage for channel_test_mmq"
```

---

### Task 4: Implement channel-map stage

**Files:**
- Modify: `crates/engine/examples/channel_test_mmq.rs`

- [ ] **Step 1: Add per-row stats struct and computation**

```rust
#[cfg(feature = "deltanet")]
struct RowStats {
    row: usize,
    max_err: f32,
    mean_err: f32,
}

/// Compute per-output-row error between two [batch_size × m] tensors.
/// Returns one RowStats per row, sorted descending by max_err.
#[cfg(feature = "deltanet")]
fn per_row_diff(y_ref: &[f32], y_mmq: &[f32], m: usize, batch_size: usize) -> Vec<RowStats> {
    let mut rows: Vec<RowStats> = (0..m).map(|r| {
        let mut max_err: f32 = 0.0;
        let mut sum_err: f64 = 0.0;
        for b in 0..batch_size {
            let idx = b * m + r;
            if idx < y_ref.len() && idx < y_mmq.len() {
                let err = (y_ref[idx] - y_mmq[idx]).abs();
                max_err = max_err.max(err);
                sum_err += err as f64;
            }
        }
        let mean_err = if batch_size > 0 { (sum_err / batch_size as f64) as f32 } else { 0.0 };
        RowStats { row: r, max_err, mean_err }
    }).collect();
    rows.sort_by(|a, b| b.max_err.partial_cmp(&a.max_err).unwrap_or(std::cmp::Ordering::Equal));
    rows
}
```

- [ ] **Step 2: Add compare function that returns raw outputs (not just stats)**

```rust
/// Like compare_residual, but returns the raw f32 outputs for per-row analysis.
#[cfg(feature = "deltanet")]
fn compare_residual_raw(
    gpu: &mut rdna_compute::Gpu,
    weight: &engine::llama::WeightTensor,
    x_data: &[f32],
    m: usize,
    k: usize,
    batch_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let x_gpu = gpu.upload_f32(x_data, &[batch_size * k]).unwrap();
    let zeros = vec![0.0f32; batch_size * m];
    let y_wmma = gpu.upload_f32(&zeros, &[batch_size * m]).unwrap();
    let y_mmq = gpu.upload_f32(&zeros, &[batch_size * m]).unwrap();

    gpu.capture_mode = true;
    gpu.gemm_hfq4g256_residual_wmma(&weight.buf, &x_gpu, &y_wmma, m, k, batch_size).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let xq = gpu.ensure_q8_1_mmq_x(&x_gpu, batch_size, k).unwrap();
    gpu.gemm_hfq4g256_mmq_set_prequant(&weight.buf, xq, &y_mmq, m, k, batch_size).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.capture_mode = false;

    let ref_out = gpu.download_f32(&y_wmma).unwrap();
    let mmq_out = gpu.download_f32(&y_mmq).unwrap();

    let _ = gpu.free_tensor(x_gpu);
    let _ = gpu.free_tensor(y_wmma);
    let _ = gpu.free_tensor(y_mmq);

    (ref_out, mmq_out)
}
```

- [ ] **Step 3: Implement `channel_map`**

Replace the TODO stub:

```rust
#[cfg(feature = "deltanet")]
fn channel_map(
    gpu: &mut rdna_compute::Gpu,
    weights: &engine::qwen35::Qwen35Weights,
    config: &engine::qwen35::Qwen35Config,
    batch_size: usize,
    threshold: f32,
    filter_layer: Option<usize>,
    filter_site: Option<&str>,
) {
    use engine::qwen35::LayerWeights;

    let site = filter_site.unwrap_or("residual");
    eprintln!("\n=== STAGE 2: channel-map (site={site}, batch_size={batch_size}, threshold={threshold}) ===\n");

    let k = config.dim;

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        if let Some(fl) = filter_layer {
            if layer_idx != fl { continue; }
        }

        // Get the weight tensor for the requested site
        let wt: Option<&engine::llama::WeightTensor> = match (layer, site) {
            (LayerWeights::DeltaNet(l), "residual") => Some(&l.wo),
            (LayerWeights::DeltaNet(l), "qkvza.qkv") => Some(&l.wqkv),
            (LayerWeights::DeltaNet(l), "qkvza.z") => Some(&l.wz),
            (LayerWeights::DeltaNet(l), "qkvza.beta") => Some(&l.w_beta),
            (LayerWeights::DeltaNet(l), "qkvza.alpha") => Some(&l.w_alpha),
            (LayerWeights::DeltaNet(l), "gate_up.gate") => Some(&l.w_gate),
            (LayerWeights::DeltaNet(l), "gate_up.up") => Some(&l.w_up),
            (LayerWeights::FullAttn(l), "residual") => Some(&l.wo),
            (LayerWeights::FullAttn(l), "qkv.q") => Some(&l.wq),
            (LayerWeights::FullAttn(l), "qkv.k") => Some(&l.wk),
            (LayerWeights::FullAttn(l), "qkv.v") => Some(&l.wv),
            (LayerWeights::FullAttn(l), "gate_up.gate") => Some(&l.w_gate),
            (LayerWeights::FullAttn(l), "gate_up.up") => Some(&l.w_up),
            (LayerWeights::DeltaNetMoe(l), "residual") => Some(&l.wo),
            (LayerWeights::DeltaNetMoe(l), "qkvza.qkv") => Some(&l.wqkv),
            (LayerWeights::DeltaNetMoe(l), "qkvza.z") => Some(&l.wz),
            (LayerWeights::DeltaNetMoe(l), "qkvza.beta") => Some(&l.w_beta),
            (LayerWeights::DeltaNetMoe(l), "qkvza.alpha") => Some(&l.w_alpha),
            (LayerWeights::FullAttnMoe(l), "residual") => Some(&l.wo),
            (LayerWeights::FullAttnMoe(l), "qkv.q") => Some(&l.wq),
            (LayerWeights::FullAttnMoe(l), "qkv.k") => Some(&l.wk),
            (LayerWeights::FullAttnMoe(l), "qkv.v") => Some(&l.wv),
            _ => None,
        };

        let wt = match wt {
            Some(w) => w,
            None => {
                eprintln!("Layer {layer_idx}: site '{site}' not available for this layer type");
                continue;
            }
        };

        let w_k = wt.k;
        let x_seed = if site == "residual" { 0x1234_5678_9ABC_DEF0u64 } else { 0xDEAD_BEEF_CAFE_BABEu64 };
        let x_data = synth_activations(batch_size, w_k, x_seed);
        let (ref_out, mmq_out) = compare_residual_raw(gpu, wt, &x_data, wt.m, w_k, batch_size);

        let rows = per_row_diff(&ref_out, &mmq_out, wt.m, batch_size);

        eprintln!("Site: {site} | Layer: {layer_idx} | Shape: {}x{}", wt.m, w_k);
        eprintln!("{:>6} | {:>10} | {:>10}", "Row", "MaxErr", "MeanErr");
        eprintln!("{}", "-".repeat(35));
        let top_n = 20.min(rows.len());
        for rs in &rows[..top_n] {
            let flag = if rs.max_err > threshold { " <<" } else { "" };
            eprintln!("{:>6} | {:>10.6} | {:>10.6}{}", rs.row, rs.max_err, rs.mean_err, flag);
        }
        let bad = rows.iter().filter(|r| r.max_err > threshold).count();
        eprintln!("  {bad}/{} rows exceed threshold {threshold}\n", wt.m);
    }
}
```

- [ ] **Step 4: Verify it compiles and runs**

Run: `cargo build --release --features deltanet --example channel_test_mmq 2>&1 | tail -3`
Expected: Compiles.

Run: `cargo run --release --features deltanet --example channel_test_mmq -- --model models/qwen3.5-9b.mq4 --stage channel-map --layer 0 --site residual 2>&1`
Expected: Prints per-row error table for layer 0's `wo` projection.

- [ ] **Step 5: Commit**

```bash
git add crates/engine/examples/channel_test_mmq.rs
git commit -m "feat(diag): implement channel-map stage for channel_test_mmq"
```

---

### Task 5: Implement layer-sweep stage

**Files:**
- Modify: `crates/engine/examples/channel_test_mmq.rs`

- [ ] **Step 1: Implement `layer_sweep`**

Replace the TODO stub:

```rust
#[cfg(feature = "deltanet")]
fn layer_sweep(
    gpu: &mut rdna_compute::Gpu,
    weights: &engine::qwen35::Qwen35Weights,
    config: &engine::qwen35::Qwen35Config,
    batch_size: usize,
    threshold: f32,
    filter_site: Option<&str>,
) {
    use engine::qwen35::LayerWeights;

    let site = filter_site.unwrap_or("residual");
    eprintln!("\n=== STAGE 3: layer-sweep (site={site}, batch_size={batch_size}, threshold={threshold}) ===\n");

    let k = config.dim;
    let mut stats: Vec<SiteStats> = Vec::new();

    eprintln!("{:>5} | {:>10} | {:>10} | {:>8}", "Layer", "MaxErr", "MeanErr", ">Thresh");
    eprintln!("{}", "-".repeat(45));

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let wt: Option<&engine::llama::WeightTensor> = match (layer, site) {
            (LayerWeights::DeltaNet(l), "residual") => Some(&l.wo),
            (LayerWeights::DeltaNet(l), "qkvza.qkv") => Some(&l.wqkv),
            (LayerWeights::DeltaNet(l), "qkvza.z") => Some(&l.wz),
            (LayerWeights::DeltaNet(l), "qkvza.beta") => Some(&l.w_beta),
            (LayerWeights::DeltaNet(l), "qkvza.alpha") => Some(&l.w_alpha),
            (LayerWeights::DeltaNet(l), "gate_up.gate") => Some(&l.w_gate),
            (LayerWeights::DeltaNet(l), "gate_up.up") => Some(&l.w_up),
            (LayerWeights::FullAttn(l), "residual") => Some(&l.wo),
            (LayerWeights::FullAttn(l), "qkv.q") => Some(&l.wq),
            (LayerWeights::FullAttn(l), "qkv.k") => Some(&l.wk),
            (LayerWeights::FullAttn(l), "qkv.v") => Some(&l.wv),
            (LayerWeights::FullAttn(l), "gate_up.gate") => Some(&l.w_gate),
            (LayerWeights::FullAttn(l), "gate_up.up") => Some(&l.w_up),
            (LayerWeights::DeltaNetMoe(l), "residual") => Some(&l.wo),
            (LayerWeights::DeltaNetMoe(l), "qkvza.qkv") => Some(&l.wqkv),
            (LayerWeights::DeltaNetMoe(l), "qkvza.z") => Some(&l.wz),
            (LayerWeights::DeltaNetMoe(l), "qkvza.beta") => Some(&l.w_beta),
            (LayerWeights::DeltaNetMoe(l), "qkvza.alpha") => Some(&l.w_alpha),
            (LayerWeights::FullAttnMoe(l), "residual") => Some(&l.wo),
            (LayerWeights::FullAttnMoe(l), "qkv.q") => Some(&l.wq),
            (LayerWeights::FullAttnMoe(l), "qkv.k") => Some(&l.wk),
            (LayerWeights::FullAttnMoe(l), "qkv.v") => Some(&l.wv),
            _ => None,
        };

        let wt = match wt {
            Some(w) => w,
            None => {
                eprintln!("{:>5} | (site '{site}' not available for this layer type)", layer_idx);
                continue;
            }
        };

        let w_k = wt.k;
        let x_seed = if site == "residual" { 0x1234_5678_9ABC_DEF0u64 } else { 0xDEAD_BEEF_CAFE_BABEu64 };
        let x_data = synth_activations(batch_size, w_k, x_seed);
        let s = compare_residual(gpu, wt, &x_data, wt.m, w_k, batch_size, site, layer_idx, threshold);

        let flag = if s.bad_count > 0 { " << WORST" } else { "" };
        eprintln!("{:>5} | {:>10.6} | {:>10.6} | {:>8}{}", layer_idx, s.max_err, s.mean_err, s.bad_count, flag);
        stats.push(s);
    }

    // Summary
    if let Some(worst) = stats.iter().max_by(|a, b| a.max_err.partial_cmp(&b.max_err).unwrap_or(std::cmp::Ordering::Equal)) {
        eprintln!("\nWorst layer: {} (max_err={:.6}, bad_count={})", worst.layer, worst.max_err, worst.bad_count);
    }

    let any_bad = stats.iter().any(|s| s.bad_count > 0);
    if any_bad {
        eprintln!("FAIL: {} layers exceed threshold {}", stats.iter().filter(|s| s.bad_count > 0).count(), threshold);
        std::process::exit(1);
    } else {
        eprintln!("OK: no elements exceed threshold {threshold}");
    }
}
```

- [ ] **Step 2: Verify it compiles and runs**

Run: `cargo build --release --features deltanet --example channel_test_mmq 2>&1 | tail -3`
Expected: Compiles.

Run: `cargo run --release --features deltanet --example channel_test_mmq -- --model models/qwen3.5-9b.mq4 --stage layer-sweep --site residual 2>&1`
Expected: Prints per-layer error for the `residual` site across all layers.

- [ ] **Step 3: Commit**

```bash
git add crates/engine/examples/channel_test_mmq.rs
git commit -m "feat(diag): implement layer-sweep stage for channel_test_mmq"
```

---

### Task 6: Extract the weight-lookup match into a shared helper

The `match (layer, site) => wt` block is duplicated across `site_scan`, `channel_map`,
and `layer_sweep`. Extract it into a shared function.

**Files:**
- Modify: `crates/engine/examples/channel_test_mmq.rs`

- [ ] **Step 1: Extract `get_weight_for_site`**

```rust
/// Look up a weight tensor by site name from a layer.
#[cfg(feature = "deltanet")]
fn get_weight_for_site<'a>(layer: &'a engine::qwen35::LayerWeights, site: &str) -> Option<&'a engine::llama::WeightTensor> {
    use engine::qwen35::LayerWeights;
    match (layer, site) {
        (LayerWeights::DeltaNet(l), "residual") => Some(&l.wo),
        (LayerWeights::DeltaNet(l), "qkvza.qkv") => Some(&l.wqkv),
        (LayerWeights::DeltaNet(l), "qkvza.z") => Some(&l.wz),
        (LayerWeights::DeltaNet(l), "qkvza.beta") => Some(&l.w_beta),
        (LayerWeights::DeltaNet(l), "qkvza.alpha") => Some(&l.w_alpha),
        (LayerWeights::DeltaNet(l), "gate_up.gate") => Some(&l.w_gate),
        (LayerWeights::DeltaNet(l), "gate_up.up") => Some(&l.w_up),
        (LayerWeights::FullAttn(l), "residual") => Some(&l.wo),
        (LayerWeights::FullAttn(l), "qkv.q") => Some(&l.wq),
        (LayerWeights::FullAttn(l), "qkv.k") => Some(&l.wk),
        (LayerWeights::FullAttn(l), "qkv.v") => Some(&l.wv),
        (LayerWeights::FullAttn(l), "gate_up.gate") => Some(&l.w_gate),
        (LayerWeights::FullAttn(l), "gate_up.up") => Some(&l.w_up),
        (LayerWeights::DeltaNetMoe(l), "residual") => Some(&l.wo),
        (LayerWeights::DeltaNetMoe(l), "qkvza.qkv") => Some(&l.wqkv),
        (LayerWeights::DeltaNetMoe(l), "qkvza.z") => Some(&l.wz),
        (LayerWeights::DeltaNetMoe(l), "qkvza.beta") => Some(&l.w_beta),
        (LayerWeights::DeltaNetMoe(l), "qkvza.alpha") => Some(&l.w_alpha),
        (LayerWeights::FullAttnMoe(l), "residual") => Some(&l.wo),
        (LayerWeights::FullAttnMoe(l), "qkv.q") => Some(&l.wq),
        (LayerWeights::FullAttnMoe(l), "qkv.k") => Some(&l.wk),
        (LayerWeights::FullAttnMoe(l), "qkv.v") => Some(&l.wv),
        _ => None,
    }
}
```

Then replace the duplicated match blocks in `channel_map` and `layer_sweep` with:
```rust
let wt = match get_weight_for_site(layer, site) {
    Some(w) => w,
    None => { eprintln!("..."); continue; }
};
```

And simplify `site_scan` to iterate over site names per layer type:
```rust
let sites_for_layer = match layer {
    LayerWeights::DeltaNet(_) => &["qkvza.qkv", "qkvza.z", "qkvza.beta", "qkvza.alpha", "gate_up.gate", "gate_up.up", "residual"][..],
    LayerWeights::FullAttn(_) => &["qkv.q", "qkv.k", "qkv.v", "gate_up.gate", "gate_up.up", "residual"][..],
    LayerWeights::DeltaNetMoe(_) => &["qkvza.qkv", "qkvza.z", "qkvza.beta", "qkvza.alpha", "residual"][..],
    LayerWeights::FullAttnMoe(_) => &["qkv.q", "qkv.k", "qkv.v", "residual"][..],
};
for site_name in sites_for_layer {
    let wt = get_weight_for_site(layer, site_name).unwrap();
    let x_seed = if *site_name == "residual" { 0x1234_5678_9ABC_DEF0u64 } else { 0xDEAD_BEEF_CAFE_BABEu64 };
    let x_data = synth_activations(batch_size, wt.k, x_seed);
    let s = compare_residual(gpu, wt, &x_data, wt.m, wt.k, batch_size, site_name, layer_idx, threshold);
    s.print();
    all_stats.push(s);
}
```

- [ ] **Step 2: Verify all three stages still work**

Run: `cargo run --release --features deltanet --example channel_test_mmq -- --model models/qwen3.5-9b.mq4 --stage site-scan 2>&1 | tail -15`
Expected: Same output as before.

- [ ] **Step 3: Commit**

```bash
git add crates/engine/examples/channel_test_mmq.rs
git commit -m "refactor(diag): extract get_weight_for_site helper, DRY site lookups"
```

---

### Task 7: Make `ensure_q8_1_mmq_x` and `gemm_hfq4g256_mmq_set_prequant` public

**DEPENDENCY:** This task must be completed before Task 2 Step 4 (first build).
Do Task 7 immediately after Task 2 Step 3 if the build fails with visibility errors.
If the methods are already `pub`, skip this task entirely.

**Files:**
- Modify: `crates/rdna-compute/src/dispatch.rs`

- [ ] **Step 1: Check visibility**

Run: `grep -n 'fn ensure_q8_1_mmq_x\|fn gemm_hfq4g256_mmq_set_prequant' crates/rdna-compute/src/dispatch.rs`

If the output shows `fn` (not `pub fn`), proceed. If already `pub fn`, skip this task.

- [ ] **Step 2: Change visibility**

At line 906, change:
```rust
fn ensure_q8_1_mmq_x(
```
to:
```rust
pub fn ensure_q8_1_mmq_x(
```

At line 4861, change:
```rust
fn gemm_hfq4g256_mmq_set_prequant(
```
to:
```rust
pub fn gemm_hfq4g256_mmq_set_prequant(
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build --release --features deltanet --example channel_test_mmq 2>&1 | tail -3`
Expected: Compiles without visibility errors.

- [ ] **Step 4: Commit**

```bash
git add crates/rdna-compute/src/dispatch.rs
git commit -m "fix(dispatch): make ensure_q8_1_mmq_x and mmq_set_prequant pub for diagnostic tools"
```

---

### Task 8: End-to-end validation on gfx1151

Run all three stages against a real model and confirm the binary produces actionable results.

**Files:** None (validation only)

- [ ] **Step 1: Run site-scan on 9B**

Run: `cargo run --release --features deltanet --example channel_test_mmq -- --model models/qwen3.5-9b.mq4 --stage site-scan --batch 128 2>&1 | tee /tmp/mmq_site_scan.txt`

Review the output. Record which (site, layer) pairs have the highest error.

- [ ] **Step 2: Run channel-map on the worst site/layer**

Using the worst pair from step 1 (e.g., layer 14, residual):

Run: `cargo run --release --features deltanet --example channel_test_mmq -- --model models/qwen3.5-9b.mq4 --stage channel-map --layer 14 --site residual --batch 128 2>&1 | tee /tmp/mmq_channel_map.txt`

Review: which rows have the highest error?

- [ ] **Step 3: Run layer-sweep on the worst site**

Run: `cargo run --release --features deltanet --example channel_test_mmq -- --model models/qwen3.5-9b.mq4 --stage layer-sweep --site residual --batch 128 2>&1 | tee /tmp/mmq_layer_sweep.txt`

Review: is the error concentrated in specific layers?

- [ ] **Step 4: Run site-scan on 27B (the canonical reproducer model)**

If `qwen3.6-27b.mq4` is available:

Run: `cargo run --release --features deltanet --example channel_test_mmq -- --model models/qwen3.6-27b.mq4 --stage site-scan --batch 128 2>&1 | tee /tmp/mmq_site_scan_27b.txt`

Compare error patterns between 9B and 27B.

- [ ] **Step 5: Write findings**

Create `findings/mmq-channel-test-results.md` with:
- Hardware: gfx1151 (Strix Halo)
- Models tested
- Which site(s) have the highest error
- Which rows/channels are worst
- Which layers concentrate the error
- Recommendation for fix approach (e.g., skip MMQ for specific site, raise batch threshold, etc.)

- [ ] **Step 6: Commit findings**

```bash
git add findings/mmq-channel-test-results.md
git commit -m "docs: MMQ channel-test results on gfx1151 — identifies guilty site/rows/layers"
```
