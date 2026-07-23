// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Produce portable MTP hidden-state shards from the exact deployed Qwen A3B
//! MQ4R trunk. One process owns one GPU and one deterministic input partition.

use hip_bridge::{HipError, HipResult};
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative::{KvMode, ModelSlot, ModelSlotConfig};
use hipfire_mtp_data::{
    f32_to_bf16_bits, window_starts, AtomicShardWriter, FeatureHeader, FeatureRecord,
};
use rdna_compute::{DType, Gpu};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    model: PathBuf,
    split: String,
    partition_index: usize,
    partition_count: usize,
    max_seq: usize,
    recursive_steps: usize,
    window_rows: usize,
    windows_per_record: usize,
    rows_per_shard: u64,
    target_rows: u64,
    trunk_sha256: String,
    source_manifest_sha256: String,
    producer_git_commit: String,
}

#[derive(Debug, Deserialize)]
struct SourceRow {
    id: Value,
    input_ids: Vec<u32>,
    assistant_start: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartitionState {
    schema_version: u32,
    partition_index: usize,
    partition_count: usize,
    last_source_ordinal: Option<u64>,
    next_shard_index: u64,
    source_records: u64,
    feature_records: u64,
    hidden_rows: u64,
    rejected: BTreeMap<String, u64>,
    complete: bool,
    stop_reason: Option<String>,
}

impl PartitionState {
    fn new(args: &Args) -> Self {
        Self {
            schema_version: 1,
            partition_index: args.partition_index,
            partition_count: args.partition_count,
            last_source_ordinal: None,
            next_shard_index: 0,
            source_records: 0,
            feature_records: 0,
            hidden_rows: 0,
            rejected: BTreeMap::new(),
            complete: false,
            stop_reason: None,
        }
    }

    fn reject(&mut self, reason: &str) {
        *self.rejected.entry(reason.to_string()).or_default() += 1;
    }
}

fn usage() -> &'static str {
    "qwen35_mtp_features \\\n+  --input <clean/train.jsonl> --output <feature-dir> --model <trunk.mq4r> \\\n+  --split train --partition-index <0..N-1> --partition-count <N> \\\n+  --trunk-sha256 <hex> --source-manifest-sha256 <hex> \\\n+  [--producer-git-commit <sha>] [--max-seq 4096] [--recursive-steps 3] \\\n+  [--window-rows 128] [--windows-per-record 2] \\\n+  [--rows-per-shard 262144] [--target-rows 25000000]"
}

fn parse_args() -> Result<Args, String> {
    let mut values = BTreeMap::<String, String>::new();
    let mut argv = env::args().skip(1);
    while let Some(key) = argv.next() {
        if key == "-h" || key == "--help" {
            println!("{}", usage());
            std::process::exit(0);
        }
        if !key.starts_with("--") {
            return Err(format!("unexpected positional argument {key}\n{}", usage()));
        }
        let value = argv
            .next()
            .ok_or_else(|| format!("missing value for {key}\n{}", usage()))?;
        values.insert(key, value);
    }
    let required = |key: &str| {
        values
            .get(key)
            .cloned()
            .ok_or_else(|| format!("missing {key}\n{}", usage()))
    };
    let parse_usize = |key: &str, default: usize| -> Result<usize, String> {
        match values.get(key) {
            Some(value) => value
                .parse()
                .map_err(|_| format!("{key} must be an unsigned integer")),
            None => Ok(default),
        }
    };
    let parse_u64 = |key: &str, default: u64| -> Result<u64, String> {
        match values.get(key) {
            Some(value) => value
                .parse()
                .map_err(|_| format!("{key} must be an unsigned integer")),
            None => Ok(default),
        }
    };
    let args = Args {
        input: PathBuf::from(required("--input")?),
        output: PathBuf::from(required("--output")?),
        model: PathBuf::from(required("--model")?),
        split: values
            .get("--split")
            .cloned()
            .unwrap_or_else(|| "train".into()),
        partition_index: parse_usize("--partition-index", 0)?,
        partition_count: parse_usize("--partition-count", 1)?,
        max_seq: parse_usize("--max-seq", 4096)?,
        recursive_steps: parse_usize("--recursive-steps", 3)?,
        window_rows: parse_usize("--window-rows", 128)?,
        windows_per_record: parse_usize("--windows-per-record", 2)?,
        rows_per_shard: parse_u64("--rows-per-shard", 262_144)?,
        target_rows: parse_u64("--target-rows", 25_000_000)?,
        trunk_sha256: required("--trunk-sha256")?,
        source_manifest_sha256: required("--source-manifest-sha256")?,
        producer_git_commit: values
            .get("--producer-git-commit")
            .cloned()
            .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string()),
    };
    if args.partition_count == 0 || args.partition_index >= args.partition_count {
        return Err("partition index must be smaller than non-zero partition count".into());
    }
    if args.max_seq < 2
        || args.recursive_steps == 0
        || args.window_rows == 0
        || args.windows_per_record == 0
        || args.rows_per_shard == 0
        || args.target_rows == 0
    {
        return Err("sequence, K, window, shard, and target sizes must be non-zero".into());
    }
    Ok(args)
}

fn io_error(context: &str, error: impl std::fmt::Display) -> HipError {
    HipError::new(0, &format!("{context}: {error}"))
}

fn state_path(args: &Args) -> PathBuf {
    args.output.join(format!(
        "{}-p{:03}-of{:03}.state.json",
        args.split, args.partition_index, args.partition_count
    ))
}

fn load_state(args: &Args) -> Result<PartitionState, String> {
    let path = state_path(args);
    if !path.exists() {
        return Ok(PartitionState::new(args));
    }
    let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let state: PartitionState = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    if state.schema_version != 1
        || state.partition_index != args.partition_index
        || state.partition_count != args.partition_count
    {
        return Err(format!(
            "{} belongs to a different schema or partition",
            path.display()
        ));
    }
    Ok(state)
}

fn save_state(args: &Args, state: &PartitionState) -> Result<(), String> {
    fs::create_dir_all(&args.output)
        .map_err(|error| format!("create {}: {error}", args.output.display()))?;
    let path = state_path(args);
    let partial = path.with_extension("json.partial");
    let bytes = serde_json::to_vec_pretty(state).map_err(|error| error.to_string())?;
    {
        let mut file = File::create(&partial)
            .map_err(|error| format!("create {}: {error}", partial.display()))?;
        file.write_all(&bytes)
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.sync_all())
            .map_err(|error| format!("write {}: {error}", partial.display()))?;
    }
    fs::rename(&partial, &path).map_err(|error| format!("rename {}: {error}", path.display()))?;
    Ok(())
}

fn shard_path(args: &Args, shard_index: u64) -> PathBuf {
    args.output.join(format!(
        "{}-p{:03}-of{:03}-s{:05}.rwf",
        args.split, args.partition_index, args.partition_count, shard_index
    ))
}

fn canonical_id(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "<invalid-id>".into()),
    }
}

fn main() -> HipResult<()> {
    let args = parse_args().map_err(|error| io_error("arguments", error))?;
    if !args.input.is_file() || !args.model.is_file() {
        return Err(io_error(
            "input",
            "input JSONL and deployed trunk must both be regular files",
        ));
    }
    fs::create_dir_all(&args.output).map_err(|error| io_error("create output", error))?;
    let mut state = load_state(&args).map_err(|error| io_error("resume state", error))?;
    if state.complete {
        eprintln!(
            "[mtp-features] partition {}/{} already complete: {} rows",
            args.partition_index, args.partition_count, state.hidden_rows
        );
        return Ok(());
    }

    // Reuse the large sequence scratch rather than paying ~25 allocations for
    // every trajectory. This is a producer-only process, before config init.
    if env::var_os("HIPFIRE_PREFILL_REUSE_PBS").is_none() {
        env::set_var("HIPFIRE_PREFILL_REUSE_PBS", "1");
    }
    if env::var_os("HIPFIRE_PREFILL_MAX_BATCH").is_none() {
        env::set_var("HIPFIRE_PREFILL_MAX_BATCH", "256");
    }

    eprintln!(
        "[mtp-features] partition {}/{} input={} model={} resume_after={:?}",
        args.partition_index,
        args.partition_count,
        args.input.display(),
        args.model.display(),
        state.last_source_ordinal
    );
    let mut gpu = Gpu::init()?;
    eprintln!("[mtp-features] GPU {}", gpu.arch);
    let slot_config = ModelSlotConfig {
        max_seq: args.max_seq,
        kv_mode: KvMode::Q8,
        repeat_window: 128,
        state_quant: qwen35::StateQuant::Q8,
    };
    let mut slot = ModelSlot::load(&mut gpu, &args.model, "mtp-feature-trunk", slot_config)?;
    let hidden_dim = slot.config.dim;
    let hidden_out = gpu.zeros(&[args.max_seq * hidden_dim], DType::F32)?;
    let header = FeatureHeader {
        schema_version: 1,
        architecture: "qwen3.5-a3b".into(),
        model: "Qwen/Qwen3.6-35B-A3B".into(),
        trunk_path: args.model.display().to_string(),
        trunk_sha256: args.trunk_sha256.clone(),
        source_manifest_sha256: args.source_manifest_sha256.clone(),
        producer_git_commit: args.producer_git_commit.clone(),
        split: args.split.clone(),
        hidden_dim: hidden_dim as u32,
        recursive_steps: args.recursive_steps as u32,
        hidden_dtype: "bf16-le".into(),
        record_checksum: "xxh3-64".into(),
        kv_mode: "q8".into(),
        state_quant: "q8".into(),
    };

    let input = BufReader::new(
        File::open(&args.input).map_err(|error| io_error("open source JSONL", error))?,
    );
    let started = Instant::now();
    let mut writer: Option<AtomicShardWriter> = None;
    let mut writer_rows = 0u64;

    for (line_index, line) in input.lines().enumerate() {
        let ordinal = line_index as u64;
        if line_index % args.partition_count != args.partition_index
            || state
                .last_source_ordinal
                .is_some_and(|last| ordinal <= last)
        {
            continue;
        }
        if state.hidden_rows >= args.target_rows {
            state.complete = true;
            state.stop_reason = Some("target_rows".into());
            break;
        }
        let line = match line {
            Ok(line) => line,
            Err(_) => {
                state.reject("read_error");
                state.last_source_ordinal = Some(ordinal);
                continue;
            }
        };
        let mut row: SourceRow = match serde_json::from_str(&line) {
            Ok(row) => row,
            Err(_) => {
                state.reject("invalid_json");
                state.last_source_ordinal = Some(ordinal);
                continue;
            }
        };
        state.source_records += 1;
        if row.input_ids.len() > args.max_seq {
            row.input_ids.truncate(args.max_seq);
        }
        if row.assistant_start >= row.input_ids.len() {
            state.reject("assistant_outside_context");
            state.last_source_ordinal = Some(ordinal);
            continue;
        }
        let completion_len = row.input_ids.len() - row.assistant_start;
        if completion_len <= args.recursive_steps {
            state.reject("insufficient_recursive_targets");
            state.last_source_ordinal = Some(ordinal);
            continue;
        }
        let available_rows = completion_len - args.recursive_steps;
        let starts = window_starts(available_rows, args.window_rows, args.windows_per_record);
        let planned_rows: u64 = starts
            .iter()
            .map(|start| available_rows.saturating_sub(*start).min(args.window_rows) as u64)
            .sum();
        if writer.is_some() && writer_rows + planned_rows > args.rows_per_shard {
            let summary = writer
                .take()
                .unwrap()
                .finish()
                .map_err(|error| io_error("finish feature shard", error))?;
            eprintln!(
                "[mtp-features] committed {} records / {} rows to {}",
                summary.records,
                summary.hidden_rows,
                summary.path.display()
            );
            state.next_shard_index += 1;
            save_state(&args, &state).map_err(|error| io_error("save state", error))?;
            writer_rows = 0;
        }
        if writer.is_none() {
            writer = Some(
                AtomicShardWriter::create(
                    shard_path(&args, state.next_shard_index),
                    header.clone(),
                )
                .map_err(|error| io_error("create feature shard", error))?,
            );
        }

        slot.dn_state.reset(&mut gpu);
        qwen35::forward_prefill_batch(
            &mut gpu,
            &slot.weights,
            &slot.config,
            &row.input_ids,
            0,
            &mut slot.kv_cache,
            &mut slot.dn_state,
            &slot.scratch,
            None,
            Some(&hidden_out),
            None,
            None,
        )?;

        let id = canonical_id(&row.id);
        for (window_index, relative_start) in starts.into_iter().enumerate() {
            if state.hidden_rows >= args.target_rows {
                break;
            }
            let rows = available_rows
                .saturating_sub(relative_start)
                .min(args.window_rows)
                .min((args.target_rows - state.hidden_rows) as usize);
            if rows == 0 {
                continue;
            }
            let absolute_start = row.assistant_start + relative_start;
            let hidden_view = hidden_out.sub_offset(absolute_start * hidden_dim, rows * hidden_dim);
            let hidden_f32 = gpu.download_f32(&hidden_view)?;
            if hidden_f32.iter().any(|value| !value.is_finite()) {
                state.reject("non_finite_hidden");
                continue;
            }
            let token_end = absolute_start + rows + args.recursive_steps;
            let record = FeatureRecord {
                id: format!("{id}#w{window_index}"),
                source_ordinal: ordinal,
                absolute_start: absolute_start as u32,
                hidden_rows: rows as u32,
                tokens: row.input_ids[absolute_start..token_end].to_vec(),
                hidden_bf16: f32_to_bf16_bits(&hidden_f32),
            };
            writer
                .as_mut()
                .unwrap()
                .write_record(&record)
                .map_err(|error| io_error("write feature record", error))?;
            writer_rows += rows as u64;
            state.feature_records += 1;
            state.hidden_rows += rows as u64;
        }
        state.last_source_ordinal = Some(ordinal);
        if state.source_records % 100 == 0 {
            let rate = state.hidden_rows as f64 / started.elapsed().as_secs_f64().max(1e-6);
            eprintln!(
                "[mtp-features] source={} records={} rows={} ({rate:.0} rows/s)",
                state.source_records, state.feature_records, state.hidden_rows
            );
        }
    }

    if let Some(active) = writer.take() {
        let summary = active
            .finish()
            .map_err(|error| io_error("finish final feature shard", error))?;
        eprintln!(
            "[mtp-features] committed {} records / {} rows to {}",
            summary.records,
            summary.hidden_rows,
            summary.path.display()
        );
        state.next_shard_index += 1;
    }
    if state.hidden_rows >= args.target_rows {
        state.complete = true;
        state.stop_reason = Some("target_rows".into());
    } else if !state.complete {
        state.complete = true;
        state.stop_reason = Some("input_exhausted".into());
    }
    save_state(&args, &state).map_err(|error| io_error("save final state", error))?;
    gpu.free_tensor(hidden_out)?;
    eprintln!(
        "[mtp-features] complete: source={} records={} rows={} reason={}",
        state.source_records,
        state.feature_records,
        state.hidden_rows,
        state.stop_reason.as_deref().unwrap_or("unknown")
    );
    Ok(())
}
