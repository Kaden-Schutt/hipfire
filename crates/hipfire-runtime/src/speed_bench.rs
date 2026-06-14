// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared speed-benchmark utilities.
//!
//! Arch examples keep their typed load/state/forward code locally. This module
//! owns the common harness pieces that should not be copied between arches.

use crate::llama::KvCache;
use rdna_compute::Gpu;
use std::path::Path;

pub struct SpeedBenchArgs {
    pub model_path: String,
    pub prefill_list: Option<Vec<usize>>,
    pub prefill_len: usize,
    pub prefill_runs: usize,
    pub gen_len: usize,
    pub warmup_len: usize,
    pub atlas_out: Option<String>,
}

impl SpeedBenchArgs {
    pub fn parse(program: &str) -> Self {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 2 {
            eprintln!(
                "Usage: {program} <model.hfq> [--prefill N] [--prefill-runs N] [--gen N] [--warmup N] [--emit-atlas <path.jsonl>]"
            );
            std::process::exit(1);
        }

        let mut parsed = Self {
            model_path: args[1].clone(),
            prefill_list: None,
            prefill_len: 32,
            prefill_runs: 1,
            gen_len: 100,
            warmup_len: 5,
            atlas_out: None,
        };

        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--prefill" => {
                    parsed.prefill_len = parse_value(&args, i, "--prefill");
                    i += 2;
                }
                "--prefill-list" => {
                    parsed.prefill_list = Some(
                        value_arg(&args, i, "--prefill-list")
                            .split(',')
                            .filter_map(|raw| raw.trim().parse::<usize>().ok())
                            .collect(),
                    );
                    i += 2;
                }
                "--prefill-runs" => {
                    parsed.prefill_runs = parse_value::<usize>(&args, i, "--prefill-runs").max(1);
                    i += 2;
                }
                "--gen" => {
                    parsed.gen_len = parse_value(&args, i, "--gen");
                    i += 2;
                }
                "--warmup" => {
                    parsed.warmup_len = parse_value(&args, i, "--warmup");
                    i += 2;
                }
                "--emit-atlas" => {
                    parsed.atlas_out = Some(value_arg(&args, i, "--emit-atlas").to_string());
                    i += 2;
                }
                other => {
                    eprintln!("unknown arg: {other}");
                    std::process::exit(1);
                }
            }
        }

        parsed
    }
}

impl SpeedBenchArgs {
    pub fn prefill_lengths(&self) -> Vec<usize> {
        if let Some(list) = &self.prefill_list {
            if !list.is_empty() {
                return list.clone();
            }
        }
        vec![self.prefill_len]
    }
}

fn value_arg<'a>(args: &'a [String], i: usize, flag: &str) -> &'a str {
    args.get(i + 1)
        .map(String::as_str)
        .unwrap_or_else(|| panic!("{flag} requires a value"))
}

fn parse_value<T>(args: &[String], i: usize, flag: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value_arg(args, i, flag)
        .parse()
        .unwrap_or_else(|err| panic!("invalid {flag} value: {err}"))
}

pub fn kv_seq_len(prefill_len: usize, warmup_len: usize, gen_len: usize) -> usize {
    (prefill_len + warmup_len + gen_len + 16).max(512)
}

pub fn kv_mode_from_env(default: &str) -> String {
    std::env::var("HIPFIRE_KV_MODE").unwrap_or_else(|_| default.to_string())
}

pub fn new_kv_cache(
    gpu: &mut Gpu,
    kv_mode: &str,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_seq: usize,
) -> Result<KvCache, String> {
    match kv_mode {
        "q8" => KvCache::new_gpu_q8(gpu, n_layers, n_kv_heads, head_dim, kv_seq)
            .map_err(|err| err.to_string()),
        "asym4" | "turbo4" => KvCache::new_gpu_asym4(gpu, n_layers, n_kv_heads, head_dim, kv_seq)
            .map_err(|err| err.to_string()),
        "asym3" | "turbo3" | "turbo" => {
            KvCache::new_gpu_asym3(gpu, n_layers, n_kv_heads, head_dim, kv_seq)
                .map_err(|err| err.to_string())
        }
        "asym2" | "turbo2" => KvCache::new_gpu_asym2(gpu, n_layers, n_kv_heads, head_dim, kv_seq)
            .map_err(|err| err.to_string()),
        other => Err(format!(
            "unknown HIPFIRE_KV_MODE: {other}  (use q8|asym4|asym3|asym2)"
        )),
    }
}

pub fn maybe_dpm_warmup(gpu: &mut Gpu, label: Option<&str>) -> Result<(), String> {
    let Ok(secs_str) = std::env::var("HIPFIRE_DPM_WARMUP_SECS") else {
        return Ok(());
    };
    let secs: f32 = secs_str.parse().unwrap_or(0.0);
    if secs > 0.0 {
        if let Some(label) = label {
            eprintln!("\n=== DPM warmup ({secs:.1}s, {label}) ===");
        }
        gpu.dpm_warmup(secs)
            .map_err(|err| format!("dpm warmup: {err}"))?;
    }
    Ok(())
}

pub fn safetensors_dir_bytes(path: &Path) -> u64 {
    std::fs::read_dir(path)
        .ok()
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                .filter_map(|e| std::fs::metadata(e.path()).ok())
                .map(|m| m.len())
                .sum::<u64>()
        })
        .unwrap_or(0)
}

pub fn file_bytes(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

pub struct LatencyStats {
    pub n: usize,
    pub sum_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub avg_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
}

impl LatencyStats {
    pub fn from_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let sum_ms: f64 = sorted.iter().sum();
        Some(Self {
            n,
            sum_ms,
            min_ms: sorted[0],
            max_ms: sorted[n - 1],
            avg_ms: sum_ms / n as f64,
            p50_ms: sorted[n / 2],
            p90_ms: sorted[(n * 90) / 100],
            p99_ms: sorted[(n.saturating_sub(1) * 99) / 100],
        })
    }
}

/// Emit the latency-class split for prefill when profiling captured the
/// per-kernel time. Returns an empty string if profiling was disabled.
pub fn split_prefill_summary(
    prefill_len: usize,
    prefill_ms: f64,
    prefill_kernel_ms: Option<f64>,
) -> String {
    if let Some(kernel_ms) = prefill_kernel_ms {
        let prefill_tok_s_kernel = prefill_len as f64 / (kernel_ms / 1000.0);
        let startup_overhead_ms = prefill_ms - kernel_ms;
        let cold_pct = if prefill_ms > 0.0 {
            (startup_overhead_ms / prefill_ms) * 100.0
        } else {
            0.0
        };
        format!(
            "  prefill_tok_s_kernel={prefill_tok_s_kernel:.1}  prefill_kernel_ms={kernel_ms:.2}  startup_overhead_ms={startup_overhead_ms:.2}  cold_overhead_pct={cold_pct:.1}"
        )
    } else {
        String::new()
    }
}
