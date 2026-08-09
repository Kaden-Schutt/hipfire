// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 3 residency proof: load parent weights onto gfx942 and report
//! measured device residency by tier against the Gate 1 projection.
//!
//! Usage:
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_residency_gate \
//!   --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
//!   [--layers 0..1] [--no-experts] [--full]
//! ```
//!
//! Defaults to `layers=0..1 load_experts=true`. `--full` loads `0..43`
//! with experts (the 150 GiB path).

use hipfire_arch_deepseek4::parent::inventory::ParentInventory;
use hipfire_arch_deepseek4::parent::weights::{ParentLoadPlan, ParentResidency, ParentWeights};
use hipfire_arch_deepseek4::parent::Ds4ParentBackend;
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;
use std::env;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Gate 1 main-tower projection (MTP excluded), GiB.
const PROJ_DENSE_BF16_GIB: f64 = 10.910;
const PROJ_EXPERT_GIB: f64 = 137.062;
const PROJ_BF16_GIB: f64 = 2.634;
const PROJ_F32_GIB: f64 = 0.132;
const PROJ_I64_GIB: f64 = 0.017;
const PROJ_TOTAL_GIB: f64 = 150.756;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::from(1)
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

    println!("=== ds4_parent_residency_gate ===");
    println!("model: {}", model_path.display());
    println!(
        "plan: layers={:?} load_experts={}",
        args.layers, args.load_experts
    );
    println!();

    let open_t0 = Instant::now();
    let source = SafetensorsSource::open(model_path).map_err(|e| {
        format!(
            "deepseek4 parent: SafetensorsSource::open({}): {e}",
            model_path.display()
        )
    })?;
    println!(
        "opened source: tensors={} ({:.1} ms)",
        source.tensor_names().len(),
        open_t0.elapsed().as_secs_f64() * 1000.0
    );

    let mut gpu = Gpu::init().map_err(|e| format!("deepseek4 parent: Gpu::init: {e:?}"))?;
    let gfx = if gpu.try_gfx942().is_some() {
        "gfx942"
    } else {
        "NOT-gfx942"
    };
    println!("gpu: {gfx}");

    let (backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    println!(
        "admit OK: num_hidden_layers={} n_routed_experts={} num_hash_layers={}",
        cfg.num_hidden_layers, cfg.n_routed_experts, cfg.num_hash_layers
    );

    let inv_t0 = Instant::now();
    let inv = ParentInventory::build(&source, &cfg)?;
    println!(
        "inventory: entries={} tensors_seen={} ({:.3} s)",
        inv.entries.len(),
        inv.totals.tensors_seen,
        inv_t0.elapsed().as_secs_f64()
    );

    let plan = ParentLoadPlan {
        layers: args.layers.clone(),
        load_experts: args.load_experts,
    };

    let load_t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    let load_secs = load_t0.elapsed().as_secs_f64();
    let res = weights.residency();

    println!();
    println!("=== measured residency ===");
    print_residency(&res);
    println!("load wall-clock: {load_secs:.3} s");
    if res.total_bytes() > 0 && load_secs > 0.0 {
        let mib_s = (res.total_bytes() as f64 / (1024.0 * 1024.0)) / load_secs;
        println!("effective device-write throughput: {mib_s:.1} MiB/s (resident bytes / wall)");
    }

    println!();
    println!("layer_range={:?} n_layers={} experts_loaded={}",
        weights.layer_range, weights.layers.len(), weights.experts_loaded);
    for layer in &weights.layers {
        println!(
            "  layer {:>2}: ratio={} compressor={} indexer={} experts={} gate_bias={} tid2eid={}",
            layer.layer_idx,
            layer.compress_ratio,
            layer.compressor.is_some(),
            layer.indexer.is_some(),
            layer.experts.len(),
            layer.gate_bias.is_some(),
            layer.tid2eid.is_some(),
        );
    }

    // Full-model comparison against Gate 1 projection.
    if plan.layers == (0..cfg.num_hidden_layers) && plan.load_experts {
        println!();
        println!("=== vs Gate 1 projection (full main tower) ===");
        compare_tier("dense_bf16", res.dense_bf16_bytes, PROJ_DENSE_BF16_GIB);
        compare_tier("expert_compressed", res.expert_compressed_bytes, PROJ_EXPERT_GIB);
        compare_tier("bf16", res.bf16_bytes, PROJ_BF16_GIB);
        compare_tier("f32", res.f32_bytes, PROJ_F32_GIB);
        compare_tier("i64", res.i64_bytes, PROJ_I64_GIB);
        compare_tier("TOTAL", res.total_bytes(), PROJ_TOTAL_GIB);
        let fits = res.total_bytes() as f64 / GIB <= 192.0;
        println!(
            "fits 192 GiB card: {fits} (headroom {:.3} GiB)",
            192.0 - res.total_bytes() as f64 / GIB
        );
    } else {
        println!();
        println!(
            "(partial load — skip full-model Gate 1 comparison; re-run with --full for that)"
        );
    }

    println!();
    println!("PASS: residency load completed");
    Ok(())
}

fn print_residency(r: &ParentResidency) {
    let row = |name: &str, b: u64| {
        println!(
            "  {name:<22} {b:>16} bytes  ({:>8.3} GiB)",
            b as f64 / GIB
        );
    };
    row("dense_bf16", r.dense_bf16_bytes);
    row("expert_compressed", r.expert_compressed_bytes);
    row("bf16", r.bf16_bytes);
    row("f32", r.f32_bytes);
    row("i64", r.i64_bytes);
    row("TOTAL", r.total_bytes());
}

fn compare_tier(name: &str, got_bytes: u64, proj_gib: f64) {
    let got_gib = got_bytes as f64 / GIB;
    let proj_bytes = (proj_gib * GIB).round() as u64;
    let rel = if proj_gib > 0.0 {
        ((got_gib - proj_gib) / proj_gib).abs()
    } else {
        0.0
    };
    let flag = if rel > 0.01 { " ** >1% DRIFT **" } else { "" };
    println!(
        "  {name:<22} got={got_gib:.3} GiB  proj={proj_gib:.3} GiB  \
         delta_bytes={:+}  rel={:.4}%{flag}",
        got_bytes as i64 - proj_bytes as i64,
        rel * 100.0
    );
}

struct Args {
    model: String,
    layers: std::ops::Range<usize>,
    load_experts: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut layers = 0..1;
    let mut load_experts = true;
    let mut argv = env::args().skip(1);
    while let Some(a) = argv.next() {
        match a.as_str() {
            "--model" => {
                model = Some(
                    argv.next()
                        .ok_or_else(|| "deepseek4 parent: --model needs a value".to_owned())?,
                );
            }
            "--layers" => {
                let v = argv
                    .next()
                    .ok_or_else(|| "deepseek4 parent: --layers needs start..end".to_owned())?;
                layers = parse_range(&v)?;
            }
            "--no-experts" => load_experts = false,
            "--full" => {
                layers = 0..43;
                load_experts = true;
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4_parent_residency_gate --model <dir> \
                     [--layers start..end] [--no-experts] [--full]"
                );
                std::process::exit(0);
            }
            other => {
                return Err(format!("deepseek4 parent: unknown arg {other}"));
            }
        }
    }
    let model = model.ok_or_else(|| {
        "usage: ds4_parent_residency_gate --model <dir> [--layers start..end] [--no-experts] [--full]"
            .to_owned()
    })?;
    Ok(Args {
        model,
        layers,
        load_experts,
    })
}

fn parse_range(s: &str) -> Result<std::ops::Range<usize>, String> {
    let (a, b) = s
        .split_once("..")
        .ok_or_else(|| format!("deepseek4 parent: bad --layers {s:?}, want start..end"))?;
    let start: usize = a
        .parse()
        .map_err(|e| format!("deepseek4 parent: bad layers start: {e}"))?;
    let end: usize = b
        .parse()
        .map_err(|e| format!("deepseek4 parent: bad layers end: {e}"))?;
    if start > end {
        return Err(format!(
            "deepseek4 parent: layers start {start} > end {end}"
        ));
    }
    Ok(start..end)
}
