// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Measured host capability profiling for eval reports.
//!
//! This runner complements static hardware buckets with direct measurements:
//! CPU memory copy bandwidth, the storage path used for `~/.hipfire/models`,
//! and optional HIP copy bandwidth when a GPU is available.

use crate::eval_harness::{collect_default_host_profile, HostProfile};
use hip_bridge::HipRuntime;
use rdna_compute::KernelCompiler;
use serde::{Deserialize, Serialize};
use std::ffi::c_void;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct HostProfileConfig {
    pub out: PathBuf,
    pub models_dir: PathBuf,
    pub size_mib: usize,
    pub storage_size_mib: usize,
    pub runs: usize,
    pub warmup_runs: usize,
    pub gpu_max_size_mib: Option<usize>,
    pub gpu_sweep_mib_step: Option<usize>,
    pub skip_gpu: bool,
    pub skip_storage: bool,
    pub json_stdout: bool,
}

impl Default for HostProfileConfig {
    fn default() -> Self {
        Self {
            out: default_output_path(),
            models_dir: default_models_dir(),
            size_mib: 128,
            storage_size_mib: 128,
            runs: 3,
            warmup_runs: 1,
            gpu_max_size_mib: None,
            gpu_sweep_mib_step: None,
            skip_gpu: false,
            skip_storage: false,
            json_stdout: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostCapabilityReport {
    pub schema: u32,
    pub kind: String,
    pub status: String,
    pub created_utc: String,
    pub runner: String,
    pub runner_version: String,
    pub build_profile: String,
    pub host_profile: HostProfile,
    pub config: HostCapabilityConfigRecord,
    pub records: Vec<BandwidthRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostCapabilityConfigRecord {
    pub models_dir: String,
    pub size_mib: usize,
    pub storage_size_mib: usize,
    pub runs: usize,
    pub warmup_runs: usize,
    pub gpu_max_size_mib: Option<usize>,
    pub gpu_sweep_mib_step: Option<usize>,
    pub gpu_free_bytes: Option<u64>,
    pub gpu_total_bytes: Option<u64>,
    pub build_profile: String,
    pub skip_gpu: bool,
    pub skip_storage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthRecord {
    pub kind: String,
    pub path: String,
    pub status: String,
    pub source: String,
    pub confidence: String,
    pub reason: Option<String>,
    pub bytes_per_run: u64,
    pub traffic_bytes_per_run: u64,
    pub runs: usize,
    pub samples_gbps: Vec<f64>,
    pub median_gbps: Option<f64>,
    pub mean_gbps: Option<f64>,
    pub min_gbps: Option<f64>,
    pub max_gbps: Option<f64>,
    pub traffic_samples_gbps: Vec<f64>,
    pub traffic_median_gbps: Option<f64>,
    pub traffic_mean_gbps: Option<f64>,
    pub traffic_min_gbps: Option<f64>,
    pub traffic_max_gbps: Option<f64>,
}

pub fn run_from_env() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print!("{}", usage());
        return Ok(());
    }
    let config = parse_args_from(args)?;
    let report = run_profile(&config);
    if let Some(parent) = config.out.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let file =
        File::create(&config.out).map_err(|e| format!("create {}: {e}", config.out.display()))?;
    serde_json::to_writer_pretty(file, &report)
        .map_err(|e| format!("write {}: {e}", config.out.display()))?;
    if config.json_stdout {
        serde_json::to_writer_pretty(std::io::stdout(), &report)
            .map_err(|e| format!("write stdout: {e}"))?;
        println!();
    } else {
        println!("{}", config.out.display());
    }
    Ok(())
}

pub fn parse_args_from<I, S>(args: I) -> Result<HostProfileConfig, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let argv: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut config = HostProfileConfig::default();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--out" => {
                config.out = PathBuf::from(take_value(&argv, i, "--out")?);
                i += 2;
            }
            "--models-dir" => {
                config.models_dir = PathBuf::from(take_value(&argv, i, "--models-dir")?);
                i += 2;
            }
            "--size-mib" => {
                config.size_mib =
                    parse_positive_usize(&take_value(&argv, i, "--size-mib")?, "--size-mib")?;
                i += 2;
            }
            "--storage-size-mib" => {
                config.storage_size_mib = parse_positive_usize(
                    &take_value(&argv, i, "--storage-size-mib")?,
                    "--storage-size-mib",
                )?;
                i += 2;
            }
            "--runs" => {
                config.runs = parse_positive_usize(&take_value(&argv, i, "--runs")?, "--runs")?;
                i += 2;
            }
            "--warmup-runs" => {
                config.warmup_runs =
                    parse_usize(&take_value(&argv, i, "--warmup-runs")?, "--warmup-runs")?;
                i += 2;
            }
            "--gpu-max-size-mib" => {
                config.gpu_max_size_mib = Some(parse_positive_usize(
                    &take_value(&argv, i, "--gpu-max-size-mib")?,
                    "--gpu-max-size-mib",
                )?);
                i += 2;
            }
            "--gpu-sweep-mib-step" => {
                config.gpu_sweep_mib_step = Some(parse_positive_usize(
                    &take_value(&argv, i, "--gpu-sweep-mib-step")?,
                    "--gpu-sweep-mib-step",
                )?);
                i += 2;
            }
            "--skip-gpu" => {
                config.skip_gpu = true;
                i += 1;
            }
            "--skip-storage" => {
                config.skip_storage = true;
                i += 1;
            }
            "--json" => {
                config.json_stdout = true;
                i += 1;
            }
            other => return Err(format!("unknown arg: {other}\n\n{}", usage())),
        }
    }
    Ok(config)
}

pub fn usage() -> String {
    "Usage:\n  hipfire-host-profile [--out <path>] [--models-dir <dir>] [--runs N]\n\n\
     Options:\n\
       --out <path>              output JSON path (default: ~/.hipfire/eval-results/host-profile/<stamp>.json)\n\
       --models-dir <dir>        model storage directory to test (default: ~/.hipfire/models)\n\
       --size-mib <N>            CPU/GPU copy test size in MiB (default: 128)\n\
       --storage-size-mib <N>    storage test size in MiB (default: 128)\n\
       --runs <N>                samples per test (default: 3)\n\
       --warmup-runs <N>         unmeasured warmup samples per test (default: 1)\n\
       --gpu-max-size-mib <N>    cap largest GPU read/write sweep payload size in MiB\n\
       --gpu-sweep-mib-step <N>  override default GPU MiB payload spacing\n\
       --skip-gpu                skip HIP copy tests\n\
       --skip-storage            skip ~/.hipfire/models storage tests\n\
       --json                    print report JSON to stdout in addition to writing --out\n"
        .to_string()
}

pub fn run_profile(config: &HostProfileConfig) -> HostCapabilityReport {
    let host_profile = collect_default_host_profile();
    let mut records = Vec::new();
    records.push(cpu_memcpy_record(
        config.size_mib,
        config.runs,
        config.warmup_runs,
    ));
    if config.skip_storage {
        records.push(skip_record(
            "storage_write_fsync",
            "models_dir",
            "storage profiling disabled by --skip-storage",
        ));
        records.push(skip_record(
            "storage_read",
            "models_dir",
            "storage profiling disabled by --skip-storage",
        ));
    } else {
        records.extend(storage_records(
            &config.models_dir,
            config.storage_size_mib,
            config.runs,
            config.warmup_runs,
        ));
    }
    let mut gpu_memory_info = None;
    if config.skip_gpu {
        records.push(skip_record(
            "gpu_host_to_device_pageable",
            "hip",
            "GPU profiling disabled by --skip-gpu",
        ));
        records.push(skip_record(
            "gpu_device_to_host_pageable",
            "hip",
            "GPU profiling disabled by --skip-gpu",
        ));
        records.push(skip_record(
            "gpu_device_write_kernel",
            "hip",
            "GPU profiling disabled by --skip-gpu",
        ));
        records.push(skip_record(
            "gpu_device_read_kernel",
            "hip",
            "GPU profiling disabled by --skip-gpu",
        ));
    } else {
        let gpu = gpu_records(
            config.size_mib,
            config.runs,
            config.warmup_runs,
            config.gpu_max_size_mib,
            config.gpu_sweep_mib_step,
        );
        gpu_memory_info = gpu.memory_info;
        records.extend(gpu.records);
    }
    let build_profile = build_profile();
    if build_profile != "release" {
        invalidate_measured_records(&mut records, &build_profile);
    }
    let status = if records.iter().any(|record| record.status == "invalid") {
        "invalid"
    } else if records.iter().any(|record| record.status == "collected") {
        "collected"
    } else {
        "not_collected"
    };
    HostCapabilityReport {
        schema: 1,
        kind: "host_capability_profile".to_string(),
        status: status.to_string(),
        created_utc: utc_now(),
        runner: "hipfire-host-profile".to_string(),
        runner_version: env!("CARGO_PKG_VERSION").to_string(),
        build_profile: build_profile.to_string(),
        host_profile,
        config: HostCapabilityConfigRecord {
            models_dir: config.models_dir.display().to_string(),
            size_mib: config.size_mib,
            storage_size_mib: config.storage_size_mib,
            runs: config.runs,
            warmup_runs: config.warmup_runs,
            gpu_max_size_mib: config.gpu_max_size_mib,
            gpu_sweep_mib_step: config.gpu_sweep_mib_step,
            gpu_free_bytes: gpu_memory_info.map(|info| info.0 as u64),
            gpu_total_bytes: gpu_memory_info.map(|info| info.1 as u64),
            build_profile: build_profile.to_string(),
            skip_gpu: config.skip_gpu,
            skip_storage: config.skip_storage,
        },
        records,
    }
}

fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn invalidate_measured_records(records: &mut [BandwidthRecord], build_profile: &str) {
    for record in records {
        if record.status == "collected" {
            record.status = "invalid".to_string();
            record.confidence = "none".to_string();
            record.reason = Some(format!(
                "bandwidth readings from {build_profile} builds are not valid evidence; rerun target/release/hipfire-host-profile"
            ));
        }
    }
}

fn cpu_memcpy_record(size_mib: usize, runs: usize, warmup_runs: usize) -> BandwidthRecord {
    let bytes = size_mib.saturating_mul(1024 * 1024);
    let src = vec![0x5au8; bytes];
    let mut dst = vec![0u8; bytes];
    for _ in 0..warmup_runs {
        dst.copy_from_slice(&src);
    }
    let mut samples = Vec::new();
    for _ in 0..runs {
        let started = Instant::now();
        dst.copy_from_slice(&src);
        let elapsed = started.elapsed().as_secs_f64();
        samples.push(gbps(bytes as u64, elapsed));
    }
    let checksum = dst.iter().fold(0u8, |acc, byte| acc ^ *byte);
    std::hint::black_box(checksum);
    measured_record(
        "cpu_memcpy",
        "system_memory",
        "std::slice::copy_from_slice",
        "medium",
        bytes as u64,
        samples,
    )
}

fn storage_records(
    models_dir: &Path,
    size_mib: usize,
    runs: usize,
    warmup_runs: usize,
) -> Vec<BandwidthRecord> {
    if let Err(err) = fs::create_dir_all(models_dir) {
        return vec![
            skip_record(
                "storage_write_fsync",
                "models_dir",
                &format!("create models dir: {err}"),
            ),
            skip_record(
                "storage_read",
                "models_dir",
                &format!("create models dir: {err}"),
            ),
        ];
    }
    let bytes = size_mib.saturating_mul(1024 * 1024);
    let temp_path = models_dir.join(format!(
        ".hipfire-host-profile-{}-{}.tmp",
        std::process::id(),
        unix_secs()
    ));
    let block = vec![0xa5u8; 1024 * 1024];
    let mut write_samples = Vec::new();
    let mut read_samples = Vec::new();
    for _ in 0..warmup_runs {
        if let Err(err) = write_temp_file_fsync(&temp_path, bytes, &block) {
            let _ = fs::remove_file(&temp_path);
            return vec![
                skip_record(
                    "storage_write_fsync",
                    &temp_path.display().to_string(),
                    &format!("warmup write/fsync failed: {err}"),
                ),
                skip_record(
                    "storage_read",
                    &temp_path.display().to_string(),
                    "storage read skipped because warmup write failed",
                ),
            ];
        }
        if let Err(err) = read_temp_file(&temp_path) {
            let _ = fs::remove_file(&temp_path);
            return vec![
                skip_record(
                    "storage_write_fsync",
                    &temp_path.display().to_string(),
                    "storage write skipped because warmup read failed",
                ),
                skip_record(
                    "storage_read",
                    &temp_path.display().to_string(),
                    &format!("warmup read failed: {err}"),
                ),
            ];
        }
    }
    for _ in 0..runs {
        let write_started = Instant::now();
        let write_result = write_temp_file_fsync(&temp_path, bytes, &block);
        let write_elapsed = write_started.elapsed().as_secs_f64();
        if let Err(err) = write_result {
            let _ = fs::remove_file(&temp_path);
            return vec![
                skip_record(
                    "storage_write_fsync",
                    &temp_path.display().to_string(),
                    &format!("write/fsync failed: {err}"),
                ),
                skip_record(
                    "storage_read",
                    &temp_path.display().to_string(),
                    "storage read skipped because write failed",
                ),
            ];
        }
        write_samples.push(gbps(bytes as u64, write_elapsed));

        let read_started = Instant::now();
        let read_result = read_temp_file(&temp_path);
        let read_elapsed = read_started.elapsed().as_secs_f64();
        if let Err(err) = read_result {
            let _ = fs::remove_file(&temp_path);
            return vec![
                measured_record(
                    "storage_write_fsync",
                    &temp_path.display().to_string(),
                    "std::fs write_all + sync_all",
                    "medium",
                    bytes as u64,
                    write_samples,
                ),
                skip_record(
                    "storage_read",
                    &temp_path.display().to_string(),
                    &format!("read failed: {err}"),
                ),
            ];
        }
        read_samples.push(gbps(bytes as u64, read_elapsed));
    }
    let _ = fs::remove_file(&temp_path);
    vec![
        measured_record(
            "storage_write_fsync",
            &temp_path.display().to_string(),
            "std::fs write_all + sync_all",
            "medium",
            bytes as u64,
            write_samples,
        ),
        measured_record(
            "storage_read",
            &temp_path.display().to_string(),
            "std::fs read",
            "medium",
            bytes as u64,
            read_samples,
        ),
    ]
}

struct GpuProfileRecords {
    records: Vec<BandwidthRecord>,
    memory_info: Option<(usize, usize)>,
}

fn gpu_records(
    size_mib: usize,
    runs: usize,
    warmup_runs: usize,
    gpu_max_size_mib: Option<usize>,
    gpu_sweep_mib_step: Option<usize>,
) -> GpuProfileRecords {
    let hip = match HipRuntime::load() {
        Ok(hip) => hip,
        Err(err) => {
            return GpuProfileRecords {
                records: vec![
                    skip_record("gpu_host_to_device_pageable", "hip", &format!("{err}")),
                    skip_record("gpu_device_to_host_pageable", "hip", &format!("{err}")),
                    skip_record("gpu_device_write_kernel", "hip", &format!("{err}")),
                    skip_record("gpu_device_read_kernel", "hip", &format!("{err}")),
                ],
                memory_info: None,
            };
        }
    };
    let memory_info = hip.get_vram_info().ok();
    let bytes = size_mib.saturating_mul(1024 * 1024);
    let src_host = vec![0x3cu8; bytes];
    let mut dst_host = vec![0u8; bytes];
    let dev_a = match hip.malloc(bytes) {
        Ok(buf) => buf,
        Err(err) => {
            return GpuProfileRecords {
                records: vec![
                    skip_record(
                        "gpu_host_to_device_pageable",
                        "hip",
                        &format!("hipMalloc: {err}"),
                    ),
                    skip_record(
                        "gpu_device_to_host_pageable",
                        "hip",
                        &format!("hipMalloc: {err}"),
                    ),
                    skip_record(
                        "gpu_device_write_kernel",
                        "hip",
                        &format!("hipMalloc: {err}"),
                    ),
                    skip_record(
                        "gpu_device_read_kernel",
                        "hip",
                        &format!("hipMalloc: {err}"),
                    ),
                ],
                memory_info,
            };
        }
    };
    let _ = hip.memcpy_htod(&dev_a, &src_host);
    let _ = hip.device_synchronize();
    for _ in 0..warmup_runs {
        if let Err(err) = hip
            .memcpy_htod(&dev_a, &src_host)
            .and_then(|_| hip.memcpy_dtoh(&mut dst_host, &dev_a))
            .and_then(|_| hip.device_synchronize())
        {
            let _ = hip.free(dev_a);
            return GpuProfileRecords {
                records: vec![
                    skip_record(
                        "gpu_host_to_device_pageable",
                        "hip",
                        &format!("warmup failed: {err}"),
                    ),
                    skip_record(
                        "gpu_device_to_host_pageable",
                        "hip",
                        &format!("warmup failed: {err}"),
                    ),
                    skip_record(
                        "gpu_device_write_kernel",
                        "hip",
                        &format!("warmup failed: {err}"),
                    ),
                    skip_record(
                        "gpu_device_read_kernel",
                        "hip",
                        &format!("warmup failed: {err}"),
                    ),
                ],
                memory_info,
            };
        }
    }

    let mut h2d = Vec::new();
    let mut d2h = Vec::new();
    for _ in 0..runs {
        let started = Instant::now();
        if let Err(err) = hip
            .memcpy_htod(&dev_a, &src_host)
            .and_then(|_| hip.device_synchronize())
        {
            let _ = hip.free(dev_a);
            return GpuProfileRecords {
                records: vec![
                    skip_record(
                        "gpu_host_to_device_pageable",
                        "hip",
                        &format!("H2D failed: {err}"),
                    ),
                    skip_record(
                        "gpu_device_to_host_pageable",
                        "hip",
                        "D2H skipped after H2D failure",
                    ),
                    skip_record(
                        "gpu_device_write_kernel",
                        "hip",
                        "GPU write sweep skipped after H2D failure",
                    ),
                    skip_record(
                        "gpu_device_read_kernel",
                        "hip",
                        "GPU read sweep skipped after H2D failure",
                    ),
                ],
                memory_info,
            };
        }
        h2d.push(gbps(bytes as u64, started.elapsed().as_secs_f64()));

        let started = Instant::now();
        if let Err(err) = hip
            .memcpy_dtoh(&mut dst_host, &dev_a)
            .and_then(|_| hip.device_synchronize())
        {
            let _ = hip.free(dev_a);
            return GpuProfileRecords {
                records: vec![
                    measured_record(
                        "gpu_host_to_device_pageable",
                        "hip",
                        "hipMemcpy H2D",
                        "medium",
                        bytes as u64,
                        h2d,
                    ),
                    skip_record(
                        "gpu_device_to_host_pageable",
                        "hip",
                        &format!("D2H failed: {err}"),
                    ),
                    skip_record(
                        "gpu_device_write_kernel",
                        "hip",
                        "GPU write sweep skipped after D2H failure",
                    ),
                    skip_record(
                        "gpu_device_read_kernel",
                        "hip",
                        "GPU read sweep skipped after D2H failure",
                    ),
                ],
                memory_info,
            };
        }
        d2h.push(gbps(bytes as u64, started.elapsed().as_secs_f64()));
    }
    let _ = hip.free(dev_a);

    let mut records = vec![
        measured_record(
            "gpu_host_to_device_pageable",
            "hip",
            "hipMemcpy H2D + hipDeviceSynchronize",
            "medium",
            bytes as u64,
            h2d,
        ),
        measured_record(
            "gpu_device_to_host_pageable",
            "hip",
            "hipMemcpy D2H + hipDeviceSynchronize",
            "medium",
            bytes as u64,
            d2h,
        ),
    ];
    records.extend(gpu_kernel_memory_records(
        &hip,
        detect_profile_arch(),
        memory_info,
        gpu_max_size_mib,
        gpu_sweep_mib_step,
        runs,
        warmup_runs,
    ));
    GpuProfileRecords {
        records,
        memory_info,
    }
}

fn gpu_kernel_memory_records(
    hip: &HipRuntime,
    arch: Option<String>,
    memory_info: Option<(usize, usize)>,
    gpu_max_size_mib: Option<usize>,
    gpu_sweep_mib_step: Option<usize>,
    runs: usize,
    warmup_runs: usize,
) -> Vec<BandwidthRecord> {
    let Some(max_bytes) = max_gpu_memory_sweep_bytes(memory_info, gpu_max_size_mib) else {
        return vec![
            skip_record(
                "gpu_device_write_kernel",
                "hip",
                "hipMemGetInfo did not provide free memory for GPU memory sweep sizing",
            ),
            skip_record(
                "gpu_device_read_kernel",
                "hip",
                "hipMemGetInfo did not provide free memory for GPU memory sweep sizing",
            ),
        ];
    };
    let Some(arch) = arch else {
        return vec![
            skip_record(
                "gpu_device_write_kernel",
                "hip",
                "GPU arch unavailable; cannot compile profiling kernels",
            ),
            skip_record(
                "gpu_device_read_kernel",
                "hip",
                "GPU arch unavailable; cannot compile profiling kernels",
            ),
        ];
    };
    let mut compiler = match KernelCompiler::new(&arch, String::new()) {
        Ok(compiler) => compiler,
        Err(err) => {
            return vec![
                skip_record(
                    "gpu_device_write_kernel",
                    "hip",
                    &format!("KernelCompiler: {err}"),
                ),
                skip_record(
                    "gpu_device_read_kernel",
                    "hip",
                    &format!("KernelCompiler: {err}"),
                ),
            ];
        }
    };
    let obj_path = match compiler.compile("host_profile_mem_bw", PROFILE_MEM_BW_KERNEL_SRC) {
        Ok(path) => path.to_path_buf(),
        Err(err) => {
            return vec![
                skip_record(
                    "gpu_device_write_kernel",
                    "hip",
                    &format!("compile profiling kernels: {err}"),
                ),
                skip_record(
                    "gpu_device_read_kernel",
                    "hip",
                    &format!("compile profiling kernels: {err}"),
                ),
            ];
        }
    };
    let module = match hip.module_load(&obj_path.display().to_string()) {
        Ok(module) => module,
        Err(err) => {
            return vec![
                skip_record(
                    "gpu_device_write_kernel",
                    "hip",
                    &format!("load profiling kernels: {err}"),
                ),
                skip_record(
                    "gpu_device_read_kernel",
                    "hip",
                    &format!("load profiling kernels: {err}"),
                ),
            ];
        }
    };
    let write_fn = match hip.module_get_function(&module, "hipfire_profile_write_u64") {
        Ok(func) => func,
        Err(err) => {
            return vec![
                skip_record(
                    "gpu_device_write_kernel",
                    "hip",
                    &format!("load write kernel: {err}"),
                ),
                skip_record(
                    "gpu_device_read_kernel",
                    "hip",
                    "read kernel skipped after write kernel load failure",
                ),
            ];
        }
    };
    let read_fn = match hip.module_get_function(&module, "hipfire_profile_read_u64") {
        Ok(func) => func,
        Err(err) => {
            return vec![
                skip_record(
                    "gpu_device_write_kernel",
                    "hip",
                    "write kernel skipped after read kernel load failure",
                ),
                skip_record(
                    "gpu_device_read_kernel",
                    "hip",
                    &format!("load read kernel: {err}"),
                ),
            ];
        }
    };
    gpu_memory_sweep_sizes(max_bytes, gpu_sweep_mib_step)
        .into_iter()
        .flat_map(|bytes| {
            gpu_kernel_memory_pair(hip, &write_fn, &read_fn, bytes, runs, warmup_runs)
        })
        .collect()
}

fn detect_profile_arch() -> Option<String> {
    collect_default_host_profile().gfx
}

fn gpu_kernel_memory_pair(
    hip: &HipRuntime,
    write_fn: &hip_bridge::Function,
    read_fn: &hip_bridge::Function,
    bytes: usize,
    runs: usize,
    warmup_runs: usize,
) -> Vec<BandwidthRecord> {
    let path = format!("hip:payload={}", size_label(bytes));
    let words = bytes / std::mem::size_of::<u64>();
    let blocks = kernel_blocks(words);
    let data = match hip.malloc(bytes) {
        Ok(buf) => buf,
        Err(err) => {
            return vec![
                skip_record(
                    "gpu_device_write_kernel",
                    &path,
                    &format!("hipMalloc data: {err}"),
                ),
                skip_record(
                    "gpu_device_read_kernel",
                    &path,
                    &format!("hipMalloc data: {err}"),
                ),
            ];
        }
    };
    let sink_bytes = blocks as usize * std::mem::size_of::<u64>();
    let sink = match hip.malloc(sink_bytes) {
        Ok(buf) => buf,
        Err(err) => {
            let _ = hip.free(data);
            return vec![
                skip_record(
                    "gpu_device_write_kernel",
                    &path,
                    &format!("hipMalloc sink: {err}"),
                ),
                skip_record(
                    "gpu_device_read_kernel",
                    &path,
                    &format!("hipMalloc sink: {err}"),
                ),
            ];
        }
    };
    if let Err(err) = hip
        .memset(&data, 0x3c, bytes)
        .and_then(|_| hip.memset(&sink, 0, sink_bytes))
        .and_then(|_| hip.device_synchronize())
    {
        let _ = hip.free(sink);
        let _ = hip.free(data);
        return vec![
            skip_record(
                "gpu_device_write_kernel",
                &path,
                &format!("warmup memset failed: {err}"),
            ),
            skip_record(
                "gpu_device_read_kernel",
                &path,
                &format!("warmup memset failed: {err}"),
            ),
        ];
    }
    for _ in 0..warmup_runs {
        if let Err(err) = launch_write_kernel(hip, write_fn, &data, words, blocks)
            .and_then(|_| launch_read_kernel(hip, read_fn, &data, &sink, words, blocks))
            .and_then(|_| hip.device_synchronize())
        {
            let _ = hip.free(sink);
            let _ = hip.free(data);
            return vec![
                skip_record(
                    "gpu_device_write_kernel",
                    &path,
                    &format!("warmup kernel failed: {err}"),
                ),
                skip_record(
                    "gpu_device_read_kernel",
                    &path,
                    &format!("warmup kernel failed: {err}"),
                ),
            ];
        }
    }
    let mut write_samples = Vec::new();
    let mut read_samples = Vec::new();
    for _ in 0..runs {
        let started = Instant::now();
        if let Err(err) = launch_write_kernel(hip, write_fn, &data, words, blocks)
            .and_then(|_| hip.device_synchronize())
        {
            let _ = hip.free(sink);
            let _ = hip.free(data);
            return vec![
                skip_record(
                    "gpu_device_write_kernel",
                    &path,
                    &format!("write kernel failed: {err}"),
                ),
                skip_record(
                    "gpu_device_read_kernel",
                    &path,
                    "read kernel skipped after write kernel failure",
                ),
            ];
        }
        write_samples.push(gbps(bytes as u64, started.elapsed().as_secs_f64()));

        let started = Instant::now();
        if let Err(err) = launch_read_kernel(hip, read_fn, &data, &sink, words, blocks)
            .and_then(|_| hip.device_synchronize())
        {
            let _ = hip.free(sink);
            let _ = hip.free(data);
            return vec![
                measured_record(
                    "gpu_device_write_kernel",
                    &path,
                    "HIP shader write bandwidth kernel",
                    "medium",
                    bytes as u64,
                    write_samples,
                ),
                skip_record(
                    "gpu_device_read_kernel",
                    &path,
                    &format!("read kernel failed: {err}"),
                ),
            ];
        }
        read_samples.push(gbps(bytes as u64, started.elapsed().as_secs_f64()));
    }
    let _ = hip.free(sink);
    let _ = hip.free(data);
    vec![
        measured_record(
            "gpu_device_write_kernel",
            &path,
            "HIP shader write bandwidth kernel",
            "medium",
            bytes as u64,
            write_samples,
        ),
        measured_record(
            "gpu_device_read_kernel",
            &path,
            "HIP shader read bandwidth kernel",
            "medium",
            bytes as u64,
            read_samples,
        ),
    ]
}

fn launch_write_kernel(
    hip: &HipRuntime,
    func: &hip_bridge::Function,
    data: &hip_bridge::DeviceBuffer,
    words: usize,
    blocks: u32,
) -> hip_bridge::HipResult<()> {
    let mut data_ptr = data.as_ptr();
    let mut words_arg = words as u64;
    let mut seed_arg = 0x9e3779b97f4a7c15u64;
    let mut params = [
        &mut data_ptr as *mut _ as *mut c_void,
        &mut words_arg as *mut _ as *mut c_void,
        &mut seed_arg as *mut _ as *mut c_void,
    ];
    unsafe { hip.launch_kernel(func, [blocks, 1, 1], [256, 1, 1], 0, None, &mut params) }
}

fn launch_read_kernel(
    hip: &HipRuntime,
    func: &hip_bridge::Function,
    data: &hip_bridge::DeviceBuffer,
    sink: &hip_bridge::DeviceBuffer,
    words: usize,
    blocks: u32,
) -> hip_bridge::HipResult<()> {
    let mut data_ptr = data.as_ptr();
    let mut sink_ptr = sink.as_ptr();
    let mut words_arg = words as u64;
    let mut params = [
        &mut data_ptr as *mut _ as *mut c_void,
        &mut sink_ptr as *mut _ as *mut c_void,
        &mut words_arg as *mut _ as *mut c_void,
    ];
    unsafe { hip.launch_kernel(func, [blocks, 1, 1], [256, 1, 1], 0, None, &mut params) }
}

fn kernel_blocks(words: usize) -> u32 {
    let blocks = (words + 255) / 256;
    blocks.clamp(1, 1024) as u32
}

const PROFILE_MEM_BW_KERNEL_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void hipfire_profile_write_u64(
    unsigned long long* __restrict__ dst,
    unsigned long long words,
    unsigned long long seed
) {
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned long long i = idx; i < words; i += stride) {
        dst[i] = seed ^ (i * 0x9e3779b97f4a7c15ULL);
    }
}

extern "C" __global__ void hipfire_profile_read_u64(
    const unsigned long long* __restrict__ src,
    unsigned long long* __restrict__ sink,
    unsigned long long words
) {
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long acc = 0;
    for (unsigned long long i = idx; i < words; i += stride) {
        acc ^= src[i];
    }
    if (threadIdx.x == 0) {
        sink[blockIdx.x] = acc;
    }
}
"#;

fn max_gpu_memory_sweep_bytes(
    memory_info: Option<(usize, usize)>,
    gpu_max_size_mib: Option<usize>,
) -> Option<usize> {
    let (free, _) = memory_info?;
    let safe_per_buffer = free.saturating_mul(80) / 100;
    let default_cap = 1024 * 1024 * 1024usize;
    let requested_cap = gpu_max_size_mib
        .map(|mib| mib.saturating_mul(1024 * 1024))
        .unwrap_or(default_cap);
    let max_bytes = requested_cap.min(safe_per_buffer);
    (max_bytes >= 1024).then_some(max_bytes)
}

fn gpu_memory_sweep_sizes(max_bytes: usize, mib_step: Option<usize>) -> Vec<usize> {
    let mut sizes: Vec<usize> = [1, 2, 4, 8, 16, 32, 64, 128]
        .into_iter()
        .map(|kib| kib * 1024)
        .collect();
    match mib_step {
        Some(step) => {
            let max_mib = max_bytes / (1024 * 1024);
            let mut mib = step;
            while mib <= max_mib {
                sizes.push(mib * 1024 * 1024);
                mib = mib.saturating_add(step);
                if mib == 0 {
                    break;
                }
            }
        }
        None => {
            sizes.extend(default_gpu_sweep_mib().map(|mib| mib * 1024 * 1024));
        }
    }
    sizes.retain(|bytes| *bytes <= max_bytes);
    if sizes.last().copied().unwrap_or(0) != max_bytes && max_bytes > 128 * 1024 {
        sizes.push(max_bytes);
    }
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

fn default_gpu_sweep_mib() -> impl Iterator<Item = usize> {
    (1..=128).chain((256..=1024).step_by(128))
}

fn size_label(bytes: usize) -> String {
    if bytes < 1024 * 1024 {
        format!("{}KiB", bytes / 1024)
    } else {
        format!("{}MiB", bytes / 1024 / 1024)
    }
}

fn write_temp_file_fsync(path: &Path, bytes: usize, block: &[u8]) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    let mut remaining = bytes;
    while remaining > 0 {
        let n = remaining.min(block.len());
        file.write_all(&block[..n])?;
        remaining -= n;
    }
    file.sync_all()?;
    Ok(())
}

fn read_temp_file(path: &Path) -> std::io::Result<u8> {
    let mut file = File::open(path)?;
    let mut buf = vec![0u8; 1024 * 1024];
    let mut checksum = 0u8;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for byte in &buf[..n] {
            checksum ^= *byte;
        }
    }
    Ok(checksum)
}

fn measured_record(
    kind: &str,
    path: &str,
    source: &str,
    confidence: &str,
    bytes_per_run: u64,
    samples_gbps: Vec<f64>,
) -> BandwidthRecord {
    measured_record_with_traffic(
        kind,
        path,
        source,
        confidence,
        bytes_per_run,
        1,
        samples_gbps,
    )
}

fn measured_record_with_traffic(
    kind: &str,
    path: &str,
    source: &str,
    confidence: &str,
    bytes_per_run: u64,
    traffic_multiplier: u64,
    samples_gbps: Vec<f64>,
) -> BandwidthRecord {
    let summary = summarize(&samples_gbps);
    let traffic_bytes_per_run = bytes_per_run.saturating_mul(traffic_multiplier);
    let traffic_samples_gbps: Vec<f64> = samples_gbps
        .iter()
        .map(|sample| sample * traffic_multiplier as f64)
        .collect();
    let traffic_summary = summarize(&traffic_samples_gbps);
    BandwidthRecord {
        kind: kind.to_string(),
        path: path.to_string(),
        status: "collected".to_string(),
        source: source.to_string(),
        confidence: confidence.to_string(),
        reason: None,
        bytes_per_run,
        traffic_bytes_per_run,
        runs: samples_gbps.len(),
        samples_gbps,
        median_gbps: summary.map(|s| s.0),
        mean_gbps: summary.map(|s| s.1),
        min_gbps: summary.map(|s| s.2),
        max_gbps: summary.map(|s| s.3),
        traffic_samples_gbps,
        traffic_median_gbps: traffic_summary.map(|s| s.0),
        traffic_mean_gbps: traffic_summary.map(|s| s.1),
        traffic_min_gbps: traffic_summary.map(|s| s.2),
        traffic_max_gbps: traffic_summary.map(|s| s.3),
    }
}

fn skip_record(kind: &str, path: &str, reason: &str) -> BandwidthRecord {
    BandwidthRecord {
        kind: kind.to_string(),
        path: path.to_string(),
        status: "skip".to_string(),
        source: "not_collected".to_string(),
        confidence: "none".to_string(),
        reason: Some(reason.to_string()),
        bytes_per_run: 0,
        traffic_bytes_per_run: 0,
        runs: 0,
        samples_gbps: Vec::new(),
        median_gbps: None,
        mean_gbps: None,
        min_gbps: None,
        max_gbps: None,
        traffic_samples_gbps: Vec::new(),
        traffic_median_gbps: None,
        traffic_mean_gbps: None,
        traffic_min_gbps: None,
        traffic_max_gbps: None,
    }
}

fn summarize(samples: &[f64]) -> Option<(f64, f64, f64, f64)> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let min = *sorted.first().unwrap();
    let max = *sorted.last().unwrap();
    Some((median, mean, min, max))
}

fn gbps(bytes: u64, seconds: f64) -> f64 {
    if seconds <= f64::EPSILON {
        return 0.0;
    }
    bytes as f64 / seconds / 1_000_000_000.0
}

fn take_value(argv: &[String], i: usize, flag: &str) -> Result<String, String> {
    argv.get(i + 1)
        .filter(|value| !value.starts_with("--"))
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_positive_usize(raw: &str, flag: &str) -> Result<usize, String> {
    let value = parse_usize(raw, flag)?;
    if value == 0 {
        Err(format!("{flag} must be a positive integer"))
    } else {
        Ok(value)
    }
}

fn parse_usize(raw: &str, flag: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|_| format!("{flag} must be a non-negative integer"))?;
    Ok(value)
}

fn default_models_dir() -> PathBuf {
    home_dir().join(".hipfire").join("models")
}

fn default_output_path() -> PathBuf {
    home_dir()
        .join(".hipfire")
        .join("eval-results")
        .join("host-profile")
        .join(format!("host-profile-{}.json", utc_stamp_compact()))
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn utc_now() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{}", unix_secs()))
}

fn utc_stamp_compact() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y%m%dT%H%M%SZ"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{}", unix_secs()))
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_host_profile_args() {
        let cfg = parse_args_from([
            "hipfire-host-profile",
            "--out",
            "/tmp/out.json",
            "--models-dir",
            "/tmp/models",
            "--size-mib",
            "4",
            "--storage-size-mib",
            "8",
            "--runs",
            "2",
            "--warmup-runs",
            "1",
            "--gpu-max-size-mib",
            "64",
            "--gpu-sweep-mib-step",
            "1",
            "--skip-gpu",
            "--skip-storage",
            "--json",
        ])
        .unwrap();
        assert_eq!(cfg.out, PathBuf::from("/tmp/out.json"));
        assert_eq!(cfg.models_dir, PathBuf::from("/tmp/models"));
        assert_eq!(cfg.size_mib, 4);
        assert_eq!(cfg.storage_size_mib, 8);
        assert_eq!(cfg.runs, 2);
        assert_eq!(cfg.warmup_runs, 1);
        assert_eq!(cfg.gpu_max_size_mib, Some(64));
        assert_eq!(cfg.gpu_sweep_mib_step, Some(1));
        assert!(cfg.skip_gpu);
        assert!(cfg.skip_storage);
        assert!(cfg.json_stdout);
    }

    #[test]
    fn summarizes_samples() {
        let summary = summarize(&[3.0, 1.0, 2.0]).unwrap();
        assert_eq!(summary.0, 2.0);
        assert_eq!(summary.1, 2.0);
        assert_eq!(summary.2, 1.0);
        assert_eq!(summary.3, 3.0);

        let even_summary = summarize(&[4.0, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(even_summary.0, 2.5);
        assert_eq!(even_summary.1, 2.5);
        assert_eq!(even_summary.2, 1.0);
        assert_eq!(even_summary.3, 4.0);
    }

    #[test]
    fn skip_profile_records_are_schemaed() {
        let mut cfg = HostProfileConfig::default();
        cfg.size_mib = 1;
        cfg.storage_size_mib = 1;
        cfg.runs = 1;
        cfg.warmup_runs = 1;
        cfg.skip_gpu = true;
        cfg.skip_storage = true;
        let report = run_profile(&cfg);
        assert_eq!(report.kind, "host_capability_profile");
        assert_eq!(report.config.warmup_runs, 1);
        assert_eq!(report.build_profile, build_profile());
        assert_eq!(report.config.build_profile, build_profile());
        assert!(report.records.iter().any(|r| r.kind == "cpu_memcpy"));
        assert!(report
            .records
            .iter()
            .any(|r| r.kind == "cpu_memcpy" && r.runs == 1));
        assert!(report
            .records
            .iter()
            .any(|r| r.kind == "gpu_device_write_kernel" && r.status == "skip"));
        assert!(report
            .records
            .iter()
            .any(|r| r.kind == "gpu_device_read_kernel" && r.status == "skip"));
    }

    #[test]
    fn gpu_memory_sweep_sizes_default_to_cache_probe_shape() {
        let sizes = gpu_memory_sweep_sizes(1024 * 1024 * 1024, None);
        for kib in [1, 2, 4, 8, 16, 32, 64, 128] {
            assert!(sizes.contains(&(kib * 1024)));
        }
        for mib in 1..=128 {
            assert!(sizes.contains(&(mib * 1024 * 1024)));
        }
        for mib in [256, 384, 512, 640, 768, 896, 1024] {
            assert!(sizes.contains(&(mib * 1024 * 1024)));
        }
        assert!(!sizes.contains(&(129 * 1024 * 1024)));
        assert!(!sizes.contains(&(255 * 1024 * 1024)));
    }

    #[test]
    fn gpu_memory_sweep_step_adds_dense_mib_payloads() {
        let sizes = gpu_memory_sweep_sizes(128 * 1024 * 1024, Some(1));
        for kib in [1, 2, 4, 8, 16, 32, 64, 128] {
            assert!(sizes.contains(&(kib * 1024)));
        }
        for mib in 1..=128 {
            assert!(sizes.contains(&(mib * 1024 * 1024)));
        }
        assert!(!sizes.contains(&(129 * 1024 * 1024)));
    }

    #[test]
    fn gpu_memory_sweep_max_respects_free_memory_and_override() {
        assert_eq!(
            max_gpu_memory_sweep_bytes(Some((100 * 1024 * 1024, 200 * 1024 * 1024)), Some(96)),
            Some(80 * 1024 * 1024)
        );
        assert_eq!(
            max_gpu_memory_sweep_bytes(Some((100 * 1024 * 1024, 200 * 1024 * 1024)), Some(16)),
            Some(16 * 1024 * 1024)
        );
        assert_eq!(max_gpu_memory_sweep_bytes(None, Some(16)), None);
    }
}
