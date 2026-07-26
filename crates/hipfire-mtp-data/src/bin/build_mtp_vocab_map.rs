// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Build a deployment vocabulary directly from HFMTPF01 training targets.
//!
//! This scanner deliberately skips the large hidden-state payload after
//! validating each record's structural lengths. The Stage 2 producer already
//! wrote and checksummed those immutable payloads; rereading roughly 400 GiB of
//! BF16 rows merely to count token ids would make vocabulary selection needlessly
//! I/O-bound. Use the existing feature audit when full checksum verification is
//! required.

use hipfire_mtp_data::{FeatureHeader, MAGIC};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;
const MAX_RECORD_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const DEFAULT_LOSS_WEIGHTS: [f64; 3] = [
    0.510_204_076_766_967_8,
    0.306_122_452_020_645_14,
    0.183_673_471_212_387_08,
];

#[derive(Debug, Deserialize)]
struct BaseVocabMap {
    source_mtp: Option<String>,
    compressed_vocab_size: usize,
    full_vocab_size: usize,
    draft_to_full: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct Coverage {
    depth: usize,
    selected_targets: u64,
    total_targets: u64,
    fraction: f64,
}

#[derive(Debug, Serialize)]
struct OutputVocabMap {
    schema_version: u32,
    source_mtp: Option<String>,
    compressed_vocab_size: usize,
    full_vocab_size: usize,
    draft_to_full: Vec<u32>,
    selection: SelectionProvenance,
}

#[derive(Debug, Serialize)]
struct SelectionProvenance {
    method: &'static str,
    alignment: &'static str,
    feature_split: String,
    feature_architecture: String,
    feature_model: String,
    feature_trunk_sha256: String,
    feature_source_manifest_sha256: String,
    feature_producer_git_commit: String,
    feature_shards: usize,
    feature_records: u64,
    usable_rows: u64,
    recursive_steps: usize,
    loss_weights: Vec<f64>,
    coverage: Vec<Coverage>,
    base_map_compressed_vocab_size: usize,
    base_map_overlap: usize,
    hidden_payload_checksums_reverified: bool,
}

struct Args {
    features: PathBuf,
    base_map: PathBuf,
    output: PathBuf,
    size: usize,
    loss_weights: Option<Vec<f64>>,
}

struct ScanSummary {
    header: FeatureHeader,
    shards: usize,
    records: u64,
    usable_rows: u64,
    counts: Vec<Vec<u64>>,
}

fn usage() -> &'static str {
    "usage: build_mtp_vocab_map --features DIR --base-map FILE --output FILE \
        [--size 16384] [--loss-weights W0,W1,W2]"
}

fn parse_args() -> Result<Args, String> {
    let mut features = None;
    let mut base_map = None;
    let mut output = None;
    let mut size = 16_384usize;
    let mut loss_weights = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Skip<env::Args>, name: &str| {
            args.next()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match arg.as_str() {
            "--features" => features = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--base-map" => base_map = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--output" => output = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--size" => {
                size = value(&mut args, &arg)?
                    .parse()
                    .map_err(|error| format!("invalid --size: {error}"))?
            }
            "--loss-weights" => {
                let parsed = value(&mut args, &arg)?
                    .split(',')
                    .map(|part| {
                        part.parse::<f64>()
                            .map_err(|error| format!("invalid loss weight {part:?}: {error}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                loss_weights = Some(parsed);
            }
            "-h" | "--help" => return Err(usage().into()),
            _ => return Err(format!("unknown argument {arg:?}\n{}", usage())),
        }
    }
    Ok(Args {
        features: features.ok_or_else(|| format!("--features is required\n{}", usage()))?,
        base_map: base_map.ok_or_else(|| format!("--base-map is required\n{}", usage()))?,
        output: output.ok_or_else(|| format!("--output is required\n{}", usage()))?,
        size,
        loss_weights,
    })
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u64_or_eof(reader: &mut impl Read) -> io::Result<Option<u64>> {
    let mut bytes = [0u8; 8];
    let mut read = 0;
    while read < bytes.len() {
        match reader.read(&mut bytes[read..])? {
            0 if read == 0 => return Ok(None),
            0 => return Err(io::Error::new(ErrorKind::UnexpectedEof, "truncated u64")),
            count => read += count,
        }
    }
    Ok(Some(u64::from_le_bytes(bytes)))
}

fn read_header(reader: &mut impl Read) -> io::Result<FeatureHeader> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid_data("invalid MTP feature shard magic"));
    }
    let header_len = read_u32(reader)? as usize;
    if header_len > MAX_HEADER_BYTES {
        return Err(invalid_data("feature header is too large"));
    }
    let mut bytes = vec![0u8; header_len];
    reader.read_exact(&mut bytes)?;
    let header: FeatureHeader =
        serde_json::from_slice(&bytes).map_err(|error| invalid_data(error.to_string()))?;
    header.validate()?;
    Ok(header)
}

fn compatible_header(reference: &FeatureHeader, candidate: &FeatureHeader) -> io::Result<()> {
    if reference.schema_version != candidate.schema_version
        || reference.architecture != candidate.architecture
        || reference.model != candidate.model
        || reference.trunk_sha256 != candidate.trunk_sha256
        || reference.source_manifest_sha256 != candidate.source_manifest_sha256
        || reference.producer_git_commit != candidate.producer_git_commit
        || reference.split != candidate.split
        || reference.hidden_dim != candidate.hidden_dim
        || reference.recursive_steps != candidate.recursive_steps
        || reference.hidden_dtype != candidate.hidden_dtype
        || reference.kv_mode != candidate.kv_mode
        || reference.state_quant != candidate.state_quant
    {
        return Err(invalid_data("feature shard header differs from reference"));
    }
    Ok(())
}

fn scan_shard(
    path: &Path,
    expected: Option<&FeatureHeader>,
    full_vocab: usize,
    counts: &mut [Vec<u64>],
) -> io::Result<(FeatureHeader, u64, u64)> {
    let mut reader = BufReader::new(File::open(path)?);
    let header = read_header(&mut reader)?;
    if let Some(reference) = expected {
        compatible_header(reference, &header)
            .map_err(|error| invalid_data(format!("{}: {error}", path.display())))?;
    }
    let k = header.recursive_steps as usize;
    if counts.len() != k {
        return Err(invalid_data("recursive-step count changed while scanning"));
    }
    let mut records = 0u64;
    let mut usable_rows = 0u64;
    while let Some(payload_len) = read_u64_or_eof(&mut reader)? {
        if payload_len > MAX_RECORD_BYTES {
            return Err(invalid_data(format!(
                "{}: feature record is too large",
                path.display()
            )));
        }
        let _checksum = read_u64(&mut reader)?;
        let id_len = read_u32(&mut reader)? as u64;
        let _source_ordinal = read_u64(&mut reader)?;
        let _absolute_start = read_u32(&mut reader)?;
        let hidden_rows = read_u32(&mut reader)? as usize;
        let token_count = read_u32(&mut reader)? as usize;
        if hidden_rows < 2 {
            return Err(invalid_data(format!(
                "{}: feature record has fewer than two hidden rows",
                path.display()
            )));
        }
        if token_count != hidden_rows + k {
            return Err(invalid_data(format!(
                "{}: token count does not equal hidden rows + K",
                path.display()
            )));
        }
        reader.seek(SeekFrom::Current(
            i64::try_from(id_len).map_err(|_| invalid_data("record id is too large"))?,
        ))?;
        let mut tokens = Vec::with_capacity(token_count);
        for _ in 0..token_count {
            let token = read_u32(&mut reader)? as usize;
            if token >= full_vocab {
                return Err(invalid_data(format!(
                    "{}: token {token} is outside full vocabulary {full_vocab}",
                    path.display()
                )));
            }
            tokens.push(token);
        }
        let effective_rows = hidden_rows - 1;
        for (depth, depth_counts) in counts.iter_mut().enumerate() {
            let start = 2 + depth;
            for &token in &tokens[start..start + effective_rows] {
                depth_counts[token] += 1;
            }
        }
        let hidden_bytes = hidden_rows
            .checked_mul(header.hidden_dim as usize)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| invalid_data("hidden payload size overflow"))?;
        let expected_payload = 24u64
            .checked_add(id_len)
            .and_then(|value| value.checked_add((token_count as u64).checked_mul(4)?))
            .and_then(|value| value.checked_add(hidden_bytes as u64))
            .ok_or_else(|| invalid_data("record payload size overflow"))?;
        if payload_len != expected_payload {
            return Err(invalid_data(format!(
                "{}: record payload length {payload_len} != structural length {expected_payload}",
                path.display()
            )));
        }
        reader.seek(SeekFrom::Current(
            i64::try_from(hidden_bytes).map_err(|_| invalid_data("hidden payload is too large"))?,
        ))?;
        records += 1;
        usable_rows += effective_rows as u64;
    }
    Ok((header, records, usable_rows))
}

fn scan_features(features: &Path, full_vocab: usize) -> io::Result<ScanSummary> {
    let mut shards = fs::read_dir(features)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|extension| extension == "rwf"))
        .collect::<Vec<_>>();
    shards.sort();
    if shards.is_empty() {
        return Err(invalid_data(format!(
            "{} contains no .rwf shards",
            features.display()
        )));
    }
    let mut header = None;
    let mut counts = Vec::new();
    let mut records = 0u64;
    let mut usable_rows = 0u64;
    for path in &shards {
        if counts.is_empty() {
            let mut reader = BufReader::new(File::open(path)?);
            let candidate = read_header(&mut reader)?;
            counts = vec![vec![0u64; full_vocab]; candidate.recursive_steps as usize];
        }
        let (candidate, shard_records, shard_rows) =
            scan_shard(path, header.as_ref(), full_vocab, &mut counts)?;
        header.get_or_insert(candidate);
        records += shard_records;
        usable_rows += shard_rows;
    }
    Ok(ScanSummary {
        header: header.expect("at least one shard"),
        shards: shards.len(),
        records,
        usable_rows,
        counts,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(invalid_data)?;
    let base: BaseVocabMap = serde_json::from_slice(&fs::read(&args.base_map)?)?;
    if base.draft_to_full.len() != base.compressed_vocab_size {
        return Err(invalid_data("base map length does not match compressed vocab size").into());
    }
    if args.size == 0 || args.size > base.full_vocab_size {
        return Err(invalid_data("--size must be within the full vocabulary").into());
    }
    let scan = scan_features(&args.features, base.full_vocab_size)?;
    let k = scan.header.recursive_steps as usize;
    let weights = args.loss_weights.unwrap_or_else(|| {
        if k == DEFAULT_LOSS_WEIGHTS.len() {
            DEFAULT_LOSS_WEIGHTS.to_vec()
        } else {
            vec![1.0 / k as f64; k]
        }
    });
    if weights.len() != k
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || weights.iter().all(|weight| *weight == 0.0)
    {
        return Err(invalid_data("--loss-weights must contain K finite nonnegative values").into());
    }
    let weight_sum: f64 = weights.iter().sum();
    let weights = weights
        .into_iter()
        .map(|weight| weight / weight_sum)
        .collect::<Vec<_>>();
    let mut ranked = (0..base.full_vocab_size)
        .map(|token| {
            let score = scan
                .counts
                .iter()
                .zip(&weights)
                .map(|(counts, weight)| counts[token] as f64 * weight)
                .sum::<f64>();
            (token as u32, score)
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|(left_token, left_score), (right_token, right_score)| {
        right_score
            .partial_cmp(left_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_token.cmp(right_token))
    });
    let draft_to_full = ranked
        .iter()
        .take(args.size)
        .map(|(token, _)| *token)
        .collect::<Vec<_>>();
    let mut selected = vec![false; base.full_vocab_size];
    for &token in &draft_to_full {
        selected[token as usize] = true;
    }
    let coverage = scan
        .counts
        .iter()
        .enumerate()
        .map(|(depth, counts)| {
            let total_targets = counts.iter().sum::<u64>();
            let selected_targets = counts
                .iter()
                .enumerate()
                .filter_map(|(token, count)| selected[token].then_some(*count))
                .sum::<u64>();
            Coverage {
                depth: depth + 1,
                selected_targets,
                total_targets,
                fraction: selected_targets as f64 / total_targets.max(1) as f64,
            }
        })
        .collect::<Vec<_>>();
    let base_selected = base
        .draft_to_full
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let base_map_overlap = draft_to_full
        .iter()
        .filter(|token| base_selected.contains(token))
        .count();
    let output = OutputVocabMap {
        schema_version: 1,
        source_mtp: base.source_mtp,
        compressed_vocab_size: args.size,
        full_vocab_size: base.full_vocab_size,
        draft_to_full,
        selection: SelectionProvenance {
            method: "runtime-shifted-weighted-target-frequency-v1",
            alignment: "runtime-shifted-v1",
            feature_split: scan.header.split.clone(),
            feature_architecture: scan.header.architecture.clone(),
            feature_model: scan.header.model.clone(),
            feature_trunk_sha256: scan.header.trunk_sha256.clone(),
            feature_source_manifest_sha256: scan.header.source_manifest_sha256.clone(),
            feature_producer_git_commit: scan.header.producer_git_commit.clone(),
            feature_shards: scan.shards,
            feature_records: scan.records,
            usable_rows: scan.usable_rows,
            recursive_steps: k,
            loss_weights: weights,
            coverage,
            base_map_compressed_vocab_size: base.compressed_vocab_size,
            base_map_overlap,
            hidden_payload_checksums_reverified: false,
        },
    };
    if args.output.exists() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            format!("refusing to overwrite {}", args.output.display()),
        )
        .into());
    }
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let partial = args.output.with_extension("json.partial");
    let mut writer = File::create(&partial)?;
    serde_json::to_writer_pretty(&mut writer, &output)?;
    writer.write_all(b"\n")?;
    writer.sync_all()?;
    fs::rename(&partial, &args.output)?;
    println!("{}", serde_json::to_string_pretty(&output.selection)?);
    Ok(())
}
