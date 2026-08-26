// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 1 of the DS4 parent-checkpoint calibration path: admit the real
//! checkpoint on gfx942 and walk all 72,317 tensors through
//! [`ParentInventory::build`].
//!
//! Usage:
//!   ds4_parent_inventory_gate --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
//!                             [--manifest out/manifest.json] \
//!                             [--skip-shard-hashes]
//!
//! `--skip-shard-hashes` is for fast iteration only. The final reported run
//! MUST hash every shard — the manifest pins the artifact.

use hipfire_ds4_parent::inventory::{ParentInventory, ParentTensorClass};
use hipfire_ds4_parent::manifest::{
    sha256_file, CaptureBoundary, CaptureInfo, ModelInfo, ModelQuantInfo,
    ParentManifest, ShardInfo, SourceInfo, MANIFEST_SCHEMA,
};
use hipfire_ds4_parent::{Ds4ParentBackend, ParentQuantConfig};
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const EXPECTED_TENSORS: usize = 72_317;
const CARD_VRAM_GIB: f64 = 192.0;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

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
            "deepseek4 parent: --model must be a safetensors directory, got {}",
            model_path.display()
        ));
    }

    println!("=== ds4_parent_inventory_gate ===");
    println!("model: {}", model_path.display());
    println!("skip_shard_hashes: {}", args.skip_shard_hashes);
    if let Some(m) = args.manifest.as_ref() {
        println!("manifest: {}", m.display());
    }
    println!();

    // 1. Open the directory the same way the deepseek4 carrier does
    //    (SafetensorsSource::open → ModelSource::Dir).
    let open_t0 = Instant::now();
    let source = SafetensorsSource::open(model_path).map_err(|e| {
        format!(
            "deepseek4 parent: SafetensorsSource::open({}): {e}",
            model_path.display()
        )
    })?;
    let open_ms = open_t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "opened SafetensorsSource: arch_id={} tensors={} ({open_ms:.1} ms)",
        ModelSource::arch_id(&source),
        source.tensor_names().len()
    );

    // 2. Admit on a real gfx942 device.
    let mut gpu = Gpu::init().map_err(|e| format!("deepseek4 parent: Gpu::init failed: {e:?}"))?;
    let gfx = gpu
        .try_gfx942()
        .map(|_| "gfx942")
        .unwrap_or("not-gfx942");
    println!("gpu: {gfx}");

    let admit_t0 = Instant::now();
    let (_backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    let admit_ms = admit_t0.elapsed().as_secs_f64() * 1000.0;
    println!("admit: OK ({admit_ms:.1} ms)");
    print_cfg(&cfg);
    check_cfg_against_contract(&cfg)?;

    // 3. Inventory walk.
    let inv_t0 = Instant::now();
    let inv = ParentInventory::build(&source, &cfg)?;
    let inv_secs = inv_t0.elapsed().as_secs_f64();
    println!();
    println!("=== inventory ===");
    println!(
        "ParentInventory::build wall-clock: {inv_secs:.3} s over {} tensors",
        inv.totals.tensors_seen
    );
    if inv_secs > 60.0 {
        println!(
            "WARNING: inventory walk is pathologically slow ({inv_secs:.1} s); \
             later gates call this on every load"
        );
    }

    inv.assert_complete(EXPECTED_TENSORS)?;
    println!("assert_complete({EXPECTED_TENSORS}): OK");

    let class_stats = collect_class_stats(&source, &inv)?;
    print_class_stats(&class_stats, &inv);

    let scale_pairings_main = inv
        .entries
        .iter()
        .filter(|e| e.scale.is_some())
        .count();
    // MTP quantized weights also had their scales claimed during build
    // (otherwise build would have refused). Count MTP weight names that
    // end in .weight among excluded_mtp.
    let scale_pairings_mtp = inv
        .excluded_mtp
        .iter()
        .filter(|n| n.ends_with(".weight"))
        .filter(|n| {
            source
                .tensor_info(n)
                .map(|t| matches!(t.dtype.as_str(), "F8_E4M3" | "I8"))
                .unwrap_or(false)
        })
        .count();
    let scale_pairings_total = scale_pairings_main + scale_pairings_mtp;
    println!();
    println!("scale pairings verified:");
    println!("  main tower: {scale_pairings_main}");
    println!("  MTP:        {scale_pairings_mtp}");
    println!("  total:      {scale_pairings_total}");

    // VRAM residency projection (main tower only).
    let vram = project_vram(&class_stats);
    print_vram(&vram);

    // 4. Evidence manifest.
    println!();
    println!("=== manifest ===");
    let (producer, engine) = ParentManifest::probe_environment("gfx942")?;
    println!(
        "engine: commit={} dirty={} rocm={}@{} arch={}",
        engine.commit,
        engine
            .dirty_diff_sha256
            .as_deref()
            .unwrap_or("clean"),
        engine.rocm_path,
        engine.rocm_version,
        engine.gpu_arch
    );

    let source_info = build_source_info(model_path, args.skip_shard_hashes)?;
    println!(
        "source: {} shards, index_sha256={}, config_sha256={}",
        source_info.shards.len(),
        &source_info.index_sha256[..16.min(source_info.index_sha256.len())],
        &source_info.config_sha256[..16.min(source_info.config_sha256.len())]
    );

    // This run consumes no corpus at all: it walks the tensor index and
    // hashes shards, nothing more. `corpus` is therefore `None` rather than a
    // zeroed CorpusInfo — an invented token count would be exactly the kind
    // of fabricated provenance field this manifest exists to prevent.
    // `validate()` permits a null corpus only when the manifest declares no
    // outputs and no captured activations, both of which hold here.
    let corpus = None;

    let produced_utc = utc_now_rfc3339();
    let manifest = ParentManifest {
        schema: MANIFEST_SCHEMA.to_string(),
        produced_utc,
        producer,
        engine,
        source: source_info,
        model: ModelInfo {
            model_type: cfg.model_type.clone(),
            num_hidden_layers: cfg.num_hidden_layers,
            mtp_loaded: false,
            rope_convention: "yarn".to_string(),
            quant: ModelQuantInfo {
                quant_method: cfg.quant_method.clone(),
                fmt: cfg.fmt.clone(),
                scale_fmt: cfg.scale_fmt.clone(),
                expert_dtype: cfg.expert_dtype.clone(),
                weight_block_size: cfg.weight_block_size,
            },
        },
        corpus,
        capture: CaptureInfo {
            boundary: CaptureBoundary::PostDynamicFp8,
            tensors: Vec::new(),
        },
        outputs: Vec::new(),
    };

    let validate_result = manifest.validate();
    match &validate_result {
        Ok(()) => println!("manifest.validate(): OK (null corpus, no outputs declared)"),
        Err(e) => return Err(format!("manifest validate failed: {e}")),
    }

    if let Some(path) = args.manifest.as_ref() {
        manifest.write_to(path)?;
        println!("wrote {}", path.display());
        // Echo a short preview so the log is self-contained.
        let written = std::fs::read_to_string(path).map_err(|e| {
            format!("deepseek4 parent: re-read manifest {}: {e}", path.display())
        })?;
        println!("--- manifest.json begin ---");
        print!("{written}");
        if !written.ends_with('\n') {
            println!();
        }
        println!("--- manifest.json end ---");
    }

    println!();
    println!("=== gate 1 summary ===");
    println!("admit:                 PASS");
    println!("assert_complete:       PASS ({EXPECTED_TENSORS})");
    println!(
        "inventory wall-clock:  {inv_secs:.3} s"
    );
    println!(
        "VRAM main-tower proj:  {:.3} GiB / {CARD_VRAM_GIB:.0} GiB (headroom {:.3} GiB) — {}",
        vram.total_gib,
        vram.headroom_gib,
        if vram.fits {
            "FITS"
        } else {
            "DOES NOT FIT"
        }
    );
    println!(
        "manifest.validate:     {}",
        if validate_result.is_ok() {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!("GATE 1: PASS");
    Ok(())
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    model: String,
    manifest: Option<PathBuf>,
    skip_shard_hashes: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut manifest: Option<PathBuf> = None;
    let mut skip_shard_hashes = false;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => {
                model = Some(
                    args.next()
                        .ok_or_else(|| "flag --model missing value".to_string())?,
                );
            }
            "--manifest" => {
                let p = args
                    .next()
                    .ok_or_else(|| "flag --manifest missing value".to_string())?;
                manifest = Some(PathBuf::from(p));
            }
            "--skip-shard-hashes" => skip_shard_hashes = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4_parent_inventory_gate --model <dir> \
                     [--manifest out/manifest.json] [--skip-shard-hashes]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }

    let model = model.ok_or_else(|| {
        "usage: ds4_parent_inventory_gate --model <dir> \
         [--manifest out/manifest.json] [--skip-shard-hashes]"
            .to_string()
    })?;
    Ok(Args {
        model,
        manifest,
        skip_shard_hashes,
    })
}

// ---------------------------------------------------------------------------
// Config checks
// ---------------------------------------------------------------------------

fn print_cfg(cfg: &ParentQuantConfig) {
    println!("ParentQuantConfig:");
    println!("  model_type:          {}", cfg.model_type);
    println!("  quant_method:        {}", cfg.quant_method);
    println!("  fmt:                 {}", cfg.fmt);
    println!("  scale_fmt:           {}", cfg.scale_fmt);
    println!("  expert_dtype:        {}", cfg.expert_dtype);
    println!("  weight_block_size:   {:?}", cfg.weight_block_size);
    println!("  num_hidden_layers:   {}", cfg.num_hidden_layers);
    println!("  num_hash_layers:     {}", cfg.num_hash_layers);
    println!("  n_routed_experts:    {}", cfg.n_routed_experts);
    println!("  num_experts_per_tok: {}", cfg.num_experts_per_tok);
    println!(
        "  compress_ratios:      len={} {:?}",
        cfg.compress_ratios.len(),
        cfg.compress_ratios
    );
}

fn check_cfg_against_contract(cfg: &ParentQuantConfig) -> Result<(), String> {
    let mut errs = Vec::new();
    if cfg.num_hidden_layers != 43 {
        errs.push(format!(
            "num_hidden_layers={} (expected 43)",
            cfg.num_hidden_layers
        ));
    }
    if cfg.num_hash_layers != 3 {
        errs.push(format!(
            "num_hash_layers={} (expected 3)",
            cfg.num_hash_layers
        ));
    }
    if cfg.n_routed_experts != 256 {
        errs.push(format!(
            "n_routed_experts={} (expected 256)",
            cfg.n_routed_experts
        ));
    }
    if cfg.num_experts_per_tok != 6 {
        errs.push(format!(
            "num_experts_per_tok={} (expected 6)",
            cfg.num_experts_per_tok
        ));
    }
    if cfg.compress_ratios.len() != 46 {
        errs.push(format!(
            "compress_ratios.len()={} (expected 46)",
            cfg.compress_ratios.len()
        ));
    }
    if !errs.is_empty() {
        return Err(format!(
            "deepseek4 parent: admitted config mismatches real checkpoint contract: {}",
            errs.join("; ")
        ));
    }
    println!("config contract check: OK (43 layers, 3 hash, 256 experts, top-k 6, 46 compress_ratios)");
    Ok(())
}

// ---------------------------------------------------------------------------
// Class stats + VRAM projection
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Debug)]
struct ClassBucket {
    count: usize,
    /// Stored payload bytes for this class (weights only for quantized tiers).
    weight_bytes: u64,
    /// Scale companion bytes (DenseFp8 / ExpertFp4 only).
    scale_bytes: u64,
}

#[derive(Default, Clone, Debug)]
struct ClassStats {
    main: [ClassBucket; 5],
    mtp: [ClassBucket; 5],
}

fn class_idx(c: ParentTensorClass) -> usize {
    match c {
        ParentTensorClass::DenseFp8 => 0,
        ParentTensorClass::ExpertFp4 => 1,
        ParentTensorClass::Bf16 => 2,
        ParentTensorClass::F32 => 3,
        ParentTensorClass::I64 => 4,
    }
}

fn class_name(i: usize) -> &'static str {
    match i {
        0 => "DenseFp8",
        1 => "ExpertFp4",
        2 => "Bf16",
        3 => "F32",
        4 => "I64",
        _ => "?",
    }
}

fn collect_class_stats(
    source: &SafetensorsSource,
    inv: &ParentInventory,
) -> Result<ClassStats, String> {
    let mut stats = ClassStats::default();

    for e in &inv.entries {
        let i = class_idx(e.class);
        let info = source.tensor_info(&e.name).ok_or_else(|| {
            format!(
                "deepseek4 parent: missing tensor_info for inventory entry {}",
                e.name
            )
        })?;
        stats.main[i].count += 1;
        stats.main[i].weight_bytes += info.data_size as u64;
        if let Some(scale) = e.scale.as_ref() {
            let sinfo = source.tensor_info(&scale.name).ok_or_else(|| {
                format!(
                    "deepseek4 parent: missing tensor_info for scale {}",
                    scale.name
                )
            })?;
            stats.main[i].scale_bytes += sinfo.data_size as u64;
        }
    }

    // MTP: excluded_mtp holds both weights and scales. Classify weights only;
    // attach scale bytes when the sibling exists.
    for name in &inv.excluded_mtp {
        if name.ends_with(".scale") {
            continue;
        }
        let info = source.tensor_info(name).ok_or_else(|| {
            format!("deepseek4 parent: missing tensor_info for MTP tensor {name}")
        })?;
        let class = match info.dtype.as_str() {
            "F8_E4M3" => ParentTensorClass::DenseFp8,
            "I8" => ParentTensorClass::ExpertFp4,
            "BF16" => ParentTensorClass::Bf16,
            "F32" => ParentTensorClass::F32,
            "I64" => ParentTensorClass::I64,
            other => {
                return Err(format!(
                    "deepseek4 parent: MTP tensor {name} has unexpected dtype {other}"
                ));
            }
        };
        let i = class_idx(class);
        stats.mtp[i].count += 1;
        stats.mtp[i].weight_bytes += info.data_size as u64;
        if matches!(
            class,
            ParentTensorClass::DenseFp8 | ParentTensorClass::ExpertFp4
        ) {
            if let Some(stem) = name.strip_suffix(".weight") {
                let sname = format!("{stem}.scale");
                if let Some(sinfo) = source.tensor_info(&sname) {
                    stats.mtp[i].scale_bytes += sinfo.data_size as u64;
                }
            }
        }
    }

    Ok(stats)
}

fn print_class_stats(stats: &ClassStats, inv: &ParentInventory) {
    println!();
    println!(
        "{:<12} {:>10} {:>14} {:>14} | {:>10} {:>14} {:>14}",
        "class",
        "main_n",
        "main_w_bytes",
        "main_s_bytes",
        "mtp_n",
        "mtp_w_bytes",
        "mtp_s_bytes"
    );
    for i in 0..5 {
        let m = &stats.main[i];
        let t = &stats.mtp[i];
        println!(
            "{:<12} {:>10} {:>14} {:>14} | {:>10} {:>14} {:>14}",
            class_name(i),
            m.count,
            m.weight_bytes,
            m.scale_bytes,
            t.count,
            t.weight_bytes,
            t.scale_bytes
        );
    }
    println!(
        "totals.payload_bytes (main weights+scales): {} ({:.3} GiB)",
        inv.totals.payload_bytes,
        inv.totals.payload_bytes as f64 / GIB
    );
    println!(
        "excluded_mtp names: {} (totals.mtp_excluded={})",
        inv.excluded_mtp.len(),
        inv.totals.mtp_excluded
    );
    println!(
        "main-tower entries: {} | tensors_seen: {}",
        inv.entries.len(),
        inv.totals.tensors_seen
    );
}

struct VramProjection {
    dense_bf16_bytes: u64,
    expert_compressed_bytes: u64,
    bf16_bytes: u64,
    f32_bytes: u64,
    i64_bytes: u64,
    total_bytes: u64,
    total_gib: f64,
    headroom_gib: f64,
    fits: bool,
}

fn project_vram(stats: &ClassStats) -> VramProjection {
    // Dense F8_E4M3 expanded to BF16 = 2× stored weight bytes.
    // Scales are consumed at dequant time and do not remain resident.
    let dense_bf16_bytes = stats.main[0].weight_bytes.saturating_mul(2);
    // Experts left compressed: I8 weights + F8_E8M0 scales as stored.
    let expert_compressed_bytes = stats.main[1]
        .weight_bytes
        .saturating_add(stats.main[1].scale_bytes);
    let bf16_bytes = stats.main[2].weight_bytes;
    let f32_bytes = stats.main[3].weight_bytes;
    let i64_bytes = stats.main[4].weight_bytes;
    let total_bytes = dense_bf16_bytes
        .saturating_add(expert_compressed_bytes)
        .saturating_add(bf16_bytes)
        .saturating_add(f32_bytes)
        .saturating_add(i64_bytes);
    let total_gib = total_bytes as f64 / GIB;
    let headroom_gib = CARD_VRAM_GIB - total_gib;
    VramProjection {
        dense_bf16_bytes,
        expert_compressed_bytes,
        bf16_bytes,
        f32_bytes,
        i64_bytes,
        total_bytes,
        total_gib,
        headroom_gib,
        fits: total_gib <= CARD_VRAM_GIB,
    }
}

fn print_vram(v: &VramProjection) {
    println!();
    println!("=== VRAM residency projection (main tower, MTP excluded) ===");
    println!(
        "  DenseFp8  F8_E4M3 → BF16:  {:>16} bytes  ({:>8.3} GiB)   [= 2 × stored F8_E4M3 weight bytes]",
        v.dense_bf16_bytes,
        v.dense_bf16_bytes as f64 / GIB
    );
    println!(
        "  ExpertFp4 I8+E8M0 stored:  {:>16} bytes  ({:>8.3} GiB)   [= I8 weights + F8_E8M0 scales]",
        v.expert_compressed_bytes,
        v.expert_compressed_bytes as f64 / GIB
    );
    println!(
        "  Bf16 as stored:            {:>16} bytes  ({:>8.3} GiB)",
        v.bf16_bytes,
        v.bf16_bytes as f64 / GIB
    );
    println!(
        "  F32 as stored:             {:>16} bytes  ({:>8.3} GiB)",
        v.f32_bytes,
        v.f32_bytes as f64 / GIB
    );
    println!(
        "  I64 as stored:             {:>16} bytes  ({:>8.3} GiB)",
        v.i64_bytes,
        v.i64_bytes as f64 / GIB
    );
    println!(
        "  ---------------------------------------------------------------"
    );
    println!(
        "  TOTAL:                     {:>16} bytes  ({:>8.3} GiB)",
        v.total_bytes, v.total_gib
    );
    println!(
        "  MI300X VRAM:               {:>16} bytes  ({:>8.3} GiB)",
        (CARD_VRAM_GIB * GIB) as u64,
        CARD_VRAM_GIB
    );
    println!(
        "  headroom:                  {:>16} bytes  ({:>8.3} GiB)",
        ((CARD_VRAM_GIB - v.total_gib).max(0.0) * GIB) as u64,
        v.headroom_gib
    );
    if v.fits {
        println!(
            "  VERDICT: parent forward FITS in {CARD_VRAM_GIB:.0} GiB with MTP excluded \
             (headroom {:.3} GiB). This is weights-only residency; activations/KV add more.",
            v.headroom_gib
        );
    } else {
        println!(
            "  VERDICT: parent forward DOES NOT FIT in {CARD_VRAM_GIB:.0} GiB with MTP excluded \
             (over by {:.3} GiB). Weights-only residency already exceeds the card.",
            -v.headroom_gib
        );
    }
}

// ---------------------------------------------------------------------------
// Manifest source pinning
// ---------------------------------------------------------------------------

fn build_source_info(root: &Path, skip_shard_hashes: bool) -> Result<SourceInfo, String> {
    let root_str = root
        .to_str()
        .ok_or_else(|| "deepseek4 parent: model root is not valid UTF-8".to_string())?
        .to_string();

    let config_path = root.join("config.json");
    let index_path = root.join("model.safetensors.index.json");
    let tokenizer_path = root.join("tokenizer.json");

    let config_sha256 = sha256_file(&config_path)?;
    let index_sha256 = if index_path.is_file() {
        sha256_file(&index_path)?
    } else {
        return Err(format!(
            "deepseek4 parent: missing index file {}",
            index_path.display()
        ));
    };
    let tokenizer_sha256 = if tokenizer_path.is_file() {
        sha256_file(&tokenizer_path)?
    } else {
        return Err(format!(
            "deepseek4 parent: missing tokenizer.json at {}",
            tokenizer_path.display()
        ));
    };

    let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(root)
        .map_err(|e| format!("deepseek4 parent: read_dir {}: {e}", root.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    shard_paths.sort();
    if shard_paths.is_empty() {
        return Err("deepseek4 parent: no .safetensors shards found".into());
    }

    let hash_t0 = Instant::now();
    let mut shards = Vec::with_capacity(shard_paths.len());
    for (i, p) in shard_paths.iter().enumerate() {
        let meta = std::fs::metadata(p).map_err(|e| {
            format!(
                "deepseek4 parent: metadata {}: {e}",
                p.display()
            )
        })?;
        let bytes = meta.len();
        let file = p
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("deepseek4 parent: bad shard name {}", p.display()))?
            .to_string();
        let sha256 = if skip_shard_hashes {
            // Non-empty placeholder so validate()'s empty-sha check does not
            // fire on a skip run; clearly marked so it cannot be mistaken for
            // a real pin. Final reported runs must not use this path.
            format!("SKIPPED_SHARD_HASH_{i:02}")
        } else {
            let t0 = Instant::now();
            let h = sha256_file(p)?;
            println!(
                "  hashed {file} ({bytes} bytes) in {:.1} s → {}",
                t0.elapsed().as_secs_f64(),
                &h[..16]
            );
            h
        };
        shards.push(ShardInfo {
            file,
            sha256,
            bytes,
        });
    }
    if !skip_shard_hashes {
        println!(
            "hashed {} shards in {:.1} s",
            shards.len(),
            hash_t0.elapsed().as_secs_f64()
        );
    } else {
        println!(
            "SKIPPED hashing {} shards (--skip-shard-hashes); placeholders are not a pin",
            shards.len()
        );
    }

    Ok(SourceInfo {
        root: root_str,
        index_sha256,
        shards,
        config_sha256,
        tokenizer_sha256,
    })
}

fn utc_now_rfc3339() -> String {
    // Prefer `date -u +%Y-%m-%dT%H:%M:%SZ` so we do not pull a time crate.
    if let Ok(out) = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    // Fallback: unix seconds as a still-sortable stamp.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}
