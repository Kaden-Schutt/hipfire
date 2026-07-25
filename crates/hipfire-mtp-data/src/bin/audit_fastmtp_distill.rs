// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Read-only integrity audit for the FastMTP prompt/completion corpus.

use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    root: PathBuf,
    expected_rows: u64,
    output: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut values = std::env::args().skip(1);
        let root = values.next().map(PathBuf::from).ok_or_else(Self::usage)?;
        let mut expected_rows = 440_000;
        let mut output = None;
        while let Some(flag) = values.next() {
            match flag.as_str() {
                "--expected-rows" => {
                    expected_rows = values
                        .next()
                        .ok_or_else(Self::usage)?
                        .parse()
                        .map_err(|_| "--expected-rows must be an integer".to_string())?;
                }
                "--output" => {
                    output = Some(PathBuf::from(values.next().ok_or_else(Self::usage)?));
                }
                "--help" | "-h" => return Err(Self::usage()),
                _ => return Err(format!("unknown argument {flag}\n{}", Self::usage())),
            }
        }
        if expected_rows == 0 {
            return Err("--expected-rows must be non-zero".to_string());
        }
        Ok(Self {
            root,
            expected_rows,
            output,
        })
    }

    fn usage() -> String {
        "usage: audit_fastmtp_distill ROOT [--expected-rows 440000] [--output AUDIT.json]"
            .to_string()
    }
}

#[derive(Debug, Deserialize)]
struct SourceManifest {
    schema_version: u32,
    target_rows: u64,
    accepted_rows: u64,
    jobs: BTreeMap<String, JobManifest>,
}

#[derive(Debug, Deserialize)]
struct JobManifest {
    rows: u64,
    sha256: String,
    sampling: Value,
}

#[derive(Debug, Deserialize)]
struct PromptRow {
    id: Value,
    tokens: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct CompletionRow {
    index: u64,
    id: Value,
    completion_tokens: Vec<u32>,
    finish_reason: Option<String>,
    sampling: Value,
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut reader = BufReader::new(
        File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?,
    );
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let read = std::io::Read::read(&mut reader, &mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn visit_jsonl<T, F>(path: &Path, mut visit: F) -> Result<(u64, String), String>
where
    T: DeserializeOwned,
    F: FnMut(u64, T) -> Result<(), String>,
{
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut raw = Vec::new();
    let mut rows = 0u64;
    let mut physical_line = 0u64;
    loop {
        raw.clear();
        let read = reader
            .read_until(b'\n', &mut raw)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        physical_line += 1;
        digest.update(&raw);
        let mut body = raw.as_slice();
        if let Some(stripped) = body.strip_suffix(b"\n") {
            body = stripped;
        }
        if let Some(stripped) = body.strip_suffix(b"\r") {
            body = stripped;
        }
        if body.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let row = serde_json::from_slice::<T>(body)
            .map_err(|error| format!("parse {} line {}: {error}", path.display(), physical_line))?;
        visit(physical_line, row)?;
        rows = rows
            .checked_add(1)
            .ok_or_else(|| format!("row count overflow in {}", path.display()))?;
    }
    Ok((rows, format!("{:x}", digest.finalize())))
}

fn write_report(path: &Path, report: &Value) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| format!("create {}: {error}", parent.display()))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("output path has no filename: {}", path.display()))?
        .to_string_lossy();
    let partial = parent.join(format!(".{file_name}.partial"));
    let mut file =
        File::create(&partial).map_err(|error| format!("create {}: {error}", partial.display()))?;
    serde_json::to_writer_pretty(&mut file, report)
        .map_err(|error| format!("write {}: {error}", partial.display()))?;
    file.write_all(b"\n")
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("finish {}: {error}", partial.display()))?;
    fs::rename(&partial, path).map_err(|error| {
        format!(
            "rename {} to {}: {error}",
            partial.display(),
            path.display()
        )
    })
}

fn audit(args: &Args) -> Result<Value, String> {
    let manifest_path = args.root.join("manifest.json");
    let manifest_bytes = fs::read(&manifest_path)
        .map_err(|error| format!("read {}: {error}", manifest_path.display()))?;
    let manifest: SourceManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("parse {}: {error}", manifest_path.display()))?;
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported source manifest schema {}",
            manifest.schema_version
        ));
    }
    let manifest_job_rows = manifest.jobs.values().try_fold(0u64, |total, job| {
        total
            .checked_add(job.rows)
            .ok_or_else(|| "manifest job row count overflow".to_string())
    })?;
    for (label, actual) in [
        ("target_rows", manifest.target_rows),
        ("accepted_rows", manifest.accepted_rows),
        ("sum(jobs.rows)", manifest_job_rows),
    ] {
        if actual != args.expected_rows {
            return Err(format!(
                "manifest {label}={actual}, expected {}",
                args.expected_rows
            ));
        }
    }
    if manifest.jobs.is_empty() {
        return Err("source manifest contains no jobs".to_string());
    }

    let completions_dir = args.root.join("completions");
    let expected_completion_files: BTreeSet<String> = manifest
        .jobs
        .keys()
        .flat_map(|job| {
            let stem = job.strip_suffix(".jsonl").unwrap_or(job);
            (0..4).map(move |gpu| format!("{stem}.gpu{gpu}.jsonl"))
        })
        .collect();
    let mut actual_completion_files = BTreeSet::new();
    for entry in fs::read_dir(&completions_dir)
        .map_err(|error| format!("read {}: {error}", completions_dir.display()))?
    {
        let entry = entry.map_err(|error| format!("read completion entry: {error}"))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".partial") {
            return Err(format!("incomplete completion artifact remains: {name}"));
        }
        if entry
            .file_type()
            .map_err(|error| format!("stat {}: {error}", entry.path().display()))?
            .is_file()
            && name.ends_with(".jsonl")
        {
            actual_completion_files.insert(name);
        }
    }
    if actual_completion_files != expected_completion_files {
        let missing = expected_completion_files
            .difference(&actual_completion_files)
            .cloned()
            .collect::<Vec<_>>();
        let unexpected = actual_completion_files
            .difference(&expected_completion_files)
            .cloned()
            .collect::<Vec<_>>();
        return Err(format!(
            "completion shard set mismatch: missing={missing:?} unexpected={unexpected:?}"
        ));
    }

    let mut total_prompt_rows = 0u64;
    let mut total_completion_rows = 0u64;
    let mut total_completion_tokens = 0u64;
    let mut finish_reasons = BTreeMap::<String, u64>::new();
    let mut gpu_rows = [0u64; 4];
    let mut gpu_tokens = [0u64; 4];
    let mut job_reports = Vec::with_capacity(manifest.jobs.len());

    for (job_name, job) in &manifest.jobs {
        let job_path = args.root.join("jobs").join(job_name);
        let mut prompt_ids = Vec::with_capacity(job.rows as usize);
        let (prompt_rows, job_sha256) = visit_jsonl::<PromptRow, _>(&job_path, |line, row| {
            if row.tokens.is_empty() {
                return Err(format!(
                    "{} line {line} has empty tokens",
                    job_path.display()
                ));
            }
            prompt_ids.push(row.id);
            Ok(())
        })?;
        if prompt_rows != job.rows {
            return Err(format!(
                "{} has {prompt_rows} rows, manifest says {}",
                job_path.display(),
                job.rows
            ));
        }
        if job_sha256 != job.sha256 {
            return Err(format!(
                "{} SHA256 {} does not match manifest {}",
                job_path.display(),
                job_sha256,
                job.sha256
            ));
        }
        total_prompt_rows = total_prompt_rows
            .checked_add(prompt_rows)
            .ok_or_else(|| "total prompt row count overflow".to_string())?;

        let stem = job_name.strip_suffix(".jsonl").unwrap_or(job_name);
        let mut seen = vec![false; job.rows as usize];
        let mut job_tokens = 0u64;
        let mut shard_reports = Vec::with_capacity(4);
        for gpu in 0..4usize {
            let completion_path = completions_dir.join(format!("{stem}.gpu{gpu}.jsonl"));
            let mut shard_tokens = 0u64;
            let (shard_rows, shard_sha256) =
                visit_jsonl::<CompletionRow, _>(&completion_path, |line, row| {
                    let index = usize::try_from(row.index).map_err(|_| {
                        format!(
                            "{} line {line} index {} does not fit usize",
                            completion_path.display(),
                            row.index
                        )
                    })?;
                    if index >= prompt_ids.len() {
                        return Err(format!(
                            "{} line {line} index {index} exceeds job rows {}",
                            completion_path.display(),
                            prompt_ids.len()
                        ));
                    }
                    if index % 4 != gpu {
                        return Err(format!(
                            "{} line {line} index {index} belongs to GPU {}, not {gpu}",
                            completion_path.display(),
                            index % 4
                        ));
                    }
                    if std::mem::replace(&mut seen[index], true) {
                        return Err(format!(
                            "{} line {line} duplicates index {index}",
                            completion_path.display()
                        ));
                    }
                    if row.id != prompt_ids[index] {
                        return Err(format!(
                            "{} line {line} id does not match prompt index {index}",
                            completion_path.display()
                        ));
                    }
                    if row.sampling != job.sampling {
                        return Err(format!(
                            "{} line {line} sampling differs from job manifest",
                            completion_path.display()
                        ));
                    }
                    if row.completion_tokens.is_empty() {
                        return Err(format!(
                            "{} line {line} has no completion tokens",
                            completion_path.display()
                        ));
                    }
                    let tokens = u64::try_from(row.completion_tokens.len())
                        .map_err(|_| "completion token count does not fit u64".to_string())?;
                    shard_tokens = shard_tokens
                        .checked_add(tokens)
                        .ok_or_else(|| "shard token count overflow".to_string())?;
                    let reason = row.finish_reason.unwrap_or_else(|| "unknown".to_string());
                    *finish_reasons.entry(reason).or_default() += 1;
                    Ok(())
                })?;
            gpu_rows[gpu] = gpu_rows[gpu]
                .checked_add(shard_rows)
                .ok_or_else(|| "GPU row count overflow".to_string())?;
            gpu_tokens[gpu] = gpu_tokens[gpu]
                .checked_add(shard_tokens)
                .ok_or_else(|| "GPU token count overflow".to_string())?;
            job_tokens = job_tokens
                .checked_add(shard_tokens)
                .ok_or_else(|| "job token count overflow".to_string())?;
            total_completion_rows = total_completion_rows
                .checked_add(shard_rows)
                .ok_or_else(|| "total completion row count overflow".to_string())?;
            total_completion_tokens = total_completion_tokens
                .checked_add(shard_tokens)
                .ok_or_else(|| "total completion token count overflow".to_string())?;
            shard_reports.push(json!({
                "gpu": gpu,
                "rows": shard_rows,
                "completion_tokens": shard_tokens,
                "sha256": shard_sha256,
            }));
        }
        if let Some(missing) = seen.iter().position(|seen| !seen) {
            return Err(format!("{job_name} has no completion for index {missing}"));
        }
        job_reports.push(json!({
            "job": job_name,
            "rows": prompt_rows,
            "completion_tokens": job_tokens,
            "job_sha256": job_sha256,
            "shards": shard_reports,
        }));
    }

    if total_prompt_rows != args.expected_rows || total_completion_rows != args.expected_rows {
        return Err(format!(
            "audited rows mismatch: prompts={total_prompt_rows} completions={total_completion_rows} expected={}",
            args.expected_rows
        ));
    }

    Ok(json!({
        "schema_version": 1,
        "pass": true,
        "root": args.root.canonicalize().unwrap_or_else(|_| args.root.clone()),
        "source_manifest_sha256": sha256_file(&manifest_path)?,
        "jobs": manifest.jobs.len(),
        "completion_shards": expected_completion_files.len(),
        "prompt_rows": total_prompt_rows,
        "completion_rows": total_completion_rows,
        "completion_tokens": total_completion_tokens,
        "per_gpu": (0..4).map(|gpu| json!({
            "gpu": gpu,
            "rows": gpu_rows[gpu],
            "completion_tokens": gpu_tokens[gpu],
        })).collect::<Vec<_>>(),
        "finish_reasons": finish_reasons,
        "per_job": job_reports,
    }))
}

fn main() -> Result<(), String> {
    let args = Args::parse()?;
    let report = audit(&args)?;
    if let Some(path) = args.output.as_deref() {
        write_report(path, &report)?;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("serialize audit report: {error}"))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

    fn fixture() -> (PathBuf, Args) {
        let serial = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!(
                "audit-fastmtp-distill-test-{}-{serial}",
                std::process::id()
            ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("jobs")).unwrap();
        fs::create_dir_all(root.join("completions")).unwrap();

        let job_name = "short-serve.jsonl";
        let mut job = String::new();
        for index in 0..4 {
            job.push_str(
                &serde_json::to_string(&json!({
                    "id": format!("row-{index}"),
                    "tokens": [1, 2, 3],
                }))
                .unwrap(),
            );
            job.push('\n');
        }
        let job_path = root.join("jobs").join(job_name);
        fs::write(&job_path, job).unwrap();
        let sampling = json!({
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        });
        let manifest = json!({
            "schema_version": 1,
            "target_rows": 4,
            "accepted_rows": 4,
            "jobs": {
                job_name: {
                    "rows": 4,
                    "sha256": sha256_file(&job_path).unwrap(),
                    "sampling": sampling,
                }
            }
        });
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        for gpu in 0..4 {
            let row = json!({
                "index": gpu,
                "id": format!("row-{gpu}"),
                "completion_tokens": [10, 11],
                "finish_reason": "stop",
                "sampling": sampling,
            });
            fs::write(
                root.join("completions")
                    .join(format!("short-serve.gpu{gpu}.jsonl")),
                format!("{}\n", serde_json::to_string(&row).unwrap()),
            )
            .unwrap();
        }
        let args = Args {
            root: root.clone(),
            expected_rows: 4,
            output: None,
        };
        (root, args)
    }

    #[test]
    fn audits_complete_partitioned_fixture() {
        let (root, args) = fixture();
        let report = audit(&args).unwrap();
        assert_eq!(report["pass"], true);
        assert_eq!(report["completion_rows"], 4);
        assert_eq!(report["completion_tokens"], 8);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_completion_in_wrong_gpu_partition() {
        let (root, args) = fixture();
        let path = root.join("completions/short-serve.gpu3.jsonl");
        let mut row: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        row["index"] = json!(0);
        fs::write(&path, format!("{}\n", serde_json::to_string(&row).unwrap())).unwrap();
        let error = audit(&args).unwrap_err();
        assert!(error.contains("belongs to GPU"), "{error}");
        fs::remove_dir_all(root).unwrap();
    }
}
