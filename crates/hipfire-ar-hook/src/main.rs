// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hipfire's narrow JSON hook boundary for Redline's Rust autoresearch loops.

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Error)]
enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid Hipfire autoresearch configuration: {0}")]
    Invalid(String),
    #[error("daemon request failed: {0}")]
    Daemon(String),
    #[error("command failed: {0}")]
    Command(String),
}

type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunKey {
    model: String,
    architecture: String,
    route: String,
    baseline: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HookConfig {
    schema_version: u32,
    run: RunKey,
    model_path: PathBuf,
    baseline_daemon: PathBuf,
    #[serde(default)]
    candidate_daemon: Option<PathBuf>,
    #[serde(default)]
    bod_path: Option<PathBuf>,
    #[serde(default)]
    environment: BTreeMap<String, String>,
    #[serde(default)]
    device_ordinal: usize,
    #[serde(default = "default_kv_mode")]
    kv_mode: String,
    #[serde(default = "default_max_seq")]
    max_seq: usize,
    #[serde(default = "default_context")]
    decode_context: usize,
    #[serde(default = "default_iterations")]
    decode_iterations: usize,
    #[serde(default = "default_warmups")]
    warmups: usize,
    #[serde(default = "default_samples")]
    samples: usize,
    #[serde(default)]
    min_lift_pct: f64,
    #[serde(default = "default_work_dir")]
    work_dir: PathBuf,
    #[serde(default)]
    serve_harness: Option<ServeHarnessConfig>,
}

fn default_kv_mode() -> String {
    "q8".into()
}
fn default_max_seq() -> usize {
    32_768
}
fn default_context() -> usize {
    128
}
fn default_iterations() -> usize {
    128
}
fn default_warmups() -> usize {
    10
}
fn default_samples() -> usize {
    5
}
fn default_work_dir() -> PathBuf {
    ".redline-work/ar/hipfire-hook".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ServeHarnessConfig {
    script: PathBuf,
    registry: PathBuf,
    #[serde(default = "default_python")]
    python: PathBuf,
    #[serde(default = "default_sampling")]
    sampling: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default)]
    session: Option<PathBuf>,
    #[serde(default)]
    prompts_file: Option<PathBuf>,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default = "default_port")]
    port: u16,
    #[serde(default = "default_timeout")]
    timeout_seconds: u64,
    #[serde(default = "default_seed")]
    seed: Option<u64>,
}

fn default_python() -> PathBuf {
    "python3".into()
}
fn default_sampling() -> String {
    "registry".into()
}
fn default_max_tokens() -> usize {
    2_048
}
fn default_mode() -> String {
    "battery".into()
}
fn default_port() -> u16 {
    11_540
}
fn default_timeout() -> u64 {
    1_800
}
fn default_seed() -> Option<u64> {
    Some(305_419_896)
}

impl HookConfig {
    fn load(path: &Path) -> Result<Self> {
        let path = fs::canonicalize(path)?;
        let mut config: Self = serde_json::from_slice(&fs::read(&path)?)?;
        if config.schema_version != SCHEMA_VERSION {
            return Err(Error::Invalid(format!(
                "config schema {}, expected {SCHEMA_VERSION}",
                config.schema_version
            )));
        }
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        resolve_path(&mut config.model_path, base);
        resolve_path(&mut config.baseline_daemon, base);
        if let Some(path) = &mut config.candidate_daemon {
            resolve_path(path, base);
        }
        if let Some(path) = &mut config.bod_path {
            resolve_path(path, base);
        }
        resolve_path(&mut config.work_dir, base);
        if let Some(harness) = &mut config.serve_harness {
            resolve_path(&mut harness.script, base);
            resolve_path(&mut harness.registry, base);
            if let Some(path) = &mut harness.session {
                resolve_path(path, base);
            }
            if let Some(path) = &mut harness.prompts_file {
                resolve_path(path, base);
            }
            if harness.python.components().count() > 1 {
                resolve_path(&mut harness.python, base);
            }
        }
        if config.samples == 0 || config.decode_iterations == 0 {
            return Err(Error::Invalid(
                "samples and decode_iterations must be non-zero".into(),
            ));
        }
        Ok(config)
    }
}

fn resolve_path(path: &mut PathBuf, base: &Path) {
    if path.is_relative() {
        *path = base.join(&*path);
    }
}

#[derive(Clone, Debug, Deserialize)]
struct CandidateTask {
    candidate: Candidate,
    #[serde(default)]
    plan: Option<CandidatePlan>,
}

#[derive(Clone, Debug, Deserialize)]
struct Candidate {
    id: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct CandidatePlan {
    #[serde(default)]
    launch: Value,
}

#[derive(Clone, Debug, Serialize)]
struct BodEntry {
    name: String,
    symbol: String,
    source: Option<PathBuf>,
    share_pct: f64,
    launches_per_iteration: u32,
    baseline_us: f64,
    shape: Value,
    capsule: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
struct Census {
    schema_version: u32,
    run: RunKey,
    generated_unix_seconds: u64,
    kernels: Vec<BodEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct LegacyBod {
    #[serde(default)]
    rows: Vec<LegacyBodRow>,
}

#[derive(Clone, Debug, Deserialize)]
struct LegacyBodRow {
    kernel: String,
    #[serde(default)]
    wall_pct: f64,
    #[serde(default)]
    n: u32,
    #[serde(default)]
    duration_us: f64,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize)]
struct RouteMetric {
    name: String,
    unit: String,
    baseline: f64,
    candidate: f64,
    lift_pct: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ModelEvaluation {
    schema_version: u32,
    candidate_id: String,
    correctness_passed: bool,
    passed: bool,
    metrics: Vec<RouteMetric>,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct Certification {
    schema_version: u32,
    candidate_id: String,
    passed: bool,
    shadow_passed: bool,
    product_ab_passed: bool,
    artifacts: BTreeMap<String, PathBuf>,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct VariantBench {
    daemon: PathBuf,
    pm4_tok_s: Vec<f64>,
    hip_tok_s: Vec<f64>,
    snapshots: Vec<Value>,
    bit_exact: bool,
    capture: Value,
}

struct Daemon {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Daemon {
    fn start(binary: &Path, config: &HookConfig, log: &Path) -> Result<Self> {
        let log = File::create(log)?;
        let mut command = Command::new(binary);
        command
            .envs(&config.environment)
            .env("HIP_VISIBLE_DEVICES", config.device_ordinal.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log));
        let mut child = command.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::Daemon("daemon stdin was not piped".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::Daemon("daemon stdout was not piped".into()))?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    fn request(&mut self, value: &Value) -> Result<Value> {
        serde_json::to_writer(&mut self.stdin, value)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let mut line = String::new();
        loop {
            line.clear();
            if self.stdout.read_line(&mut line)? == 0 {
                let status = self.child.try_wait()?;
                return Err(Error::Daemon(format!(
                    "daemon closed stdout before a response (status={status:?})"
                )));
            }
            let Ok(response) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if response.get("type").and_then(Value::as_str) == Some("error") {
                return Err(Error::Daemon(
                    response
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown daemon error")
                        .to_owned(),
                ));
            }
            return Ok(response);
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = serde_json::to_writer(&mut self.stdin, &json!({"type": "unload"}));
        let _ = self.stdin.write_all(b"\n");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let command = args
        .first()
        .map(String::as_str)
        .ok_or("usage: hipfire-ar-hook <census|model|certify> --config CONFIG.json")?;
    let config_path = take_value(&args, "--config")?;
    let config = HookConfig::load(Path::new(&config_path))?;
    fs::create_dir_all(&config.work_dir)?;
    match command {
        "census" => {
            let _input = read_stdin_json()?;
            serde_json::to_writer(std::io::stdout(), &make_census(&config)?)?;
        }
        "model" => {
            let task: CandidateTask = serde_json::from_value(read_stdin_json()?)?;
            serde_json::to_writer(std::io::stdout(), &evaluate_model(&config, &task)?)?;
        }
        "certify" => {
            let task: CandidateTask = serde_json::from_value(read_stdin_json()?)?;
            serde_json::to_writer(std::io::stdout(), &certify(&config, &task)?)?;
        }
        _ => return Err(format!("unknown hipfire-ar-hook command {command:?}").into()),
    }
    println!();
    Ok(())
}

fn take_value(args: &[String], name: &str) -> std::result::Result<String, String> {
    let index = args
        .iter()
        .position(|argument| argument == name)
        .ok_or_else(|| format!("missing {name}"))?;
    args.get(index + 1)
        .cloned()
        .ok_or_else(|| format!("missing value after {name}"))
}

fn read_stdin_json() -> Result<Value> {
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes)?;
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Ok(Value::Null);
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn make_census(config: &HookConfig) -> Result<Census> {
    let bod_path = config
        .bod_path
        .as_ref()
        .ok_or_else(|| Error::Invalid("census requires bod_path".into()))?;
    let bod: LegacyBod = serde_json::from_slice(&fs::read(bod_path)?)?;
    let kernels = bod
        .rows
        .into_iter()
        .map(|row| {
            let source = row
                .extra
                .get("source")
                .and_then(Value::as_str)
                .map(PathBuf::from);
            let capsule = row
                .extra
                .get("capsule")
                .and_then(Value::as_str)
                .map(PathBuf::from);
            let shape = Value::Object(row.extra.into_iter().collect());
            BodEntry {
                name: row.kernel.clone(),
                symbol: row.kernel,
                source,
                share_pct: row.wall_pct,
                launches_per_iteration: row.n,
                baseline_us: row.duration_us,
                shape,
                capsule,
            }
        })
        .collect();
    Ok(Census {
        schema_version: SCHEMA_VERSION,
        run: config.run.clone(),
        generated_unix_seconds: unix_seconds(),
        kernels,
    })
}

fn evaluate_model(config: &HookConfig, task: &CandidateTask) -> Result<ModelEvaluation> {
    let candidate_daemon = candidate_daemon(config, task)?;
    let id = safe_component(&task.candidate.id);
    let baseline = run_variant(
        config,
        &config.baseline_daemon,
        &format!("{id}.model-baseline"),
    )?;
    let candidate = run_variant(config, &candidate_daemon, &format!("{id}.model-candidate"))?;
    let correctness_passed = compatible_snapshots(&baseline, &candidate);
    let baseline_pm4 = median(&baseline.pm4_tok_s)?;
    let candidate_pm4 = median(&candidate.pm4_tok_s)?;
    let baseline_hip = median(&baseline.hip_tok_s)?;
    let candidate_hip = median(&candidate.hip_tok_s)?;
    let pm4_lift = lift_pct(baseline_pm4, candidate_pm4);
    let hip_lift = lift_pct(baseline_hip, candidate_hip);
    let evidence = json!({"baseline": baseline, "candidate": candidate});
    let evidence_path = evidence_path(config, &task.candidate.id, "model");
    write_json(&evidence_path, &evidence)?;
    Ok(ModelEvaluation {
        schema_version: SCHEMA_VERSION,
        candidate_id: task.candidate.id.clone(),
        correctness_passed,
        passed: correctness_passed && pm4_lift >= config.min_lift_pct,
        metrics: vec![
            RouteMetric {
                name: "retained_pm4_decode".into(),
                unit: "tok/s".into(),
                baseline: baseline_pm4,
                candidate: candidate_pm4,
                lift_pct: pm4_lift,
            },
            RouteMetric {
                name: "ordinary_hip_decode".into(),
                unit: "tok/s".into(),
                baseline: baseline_hip,
                candidate: candidate_hip,
                lift_pct: hip_lift,
            },
        ],
        notes: vec![
            format!("evidence={}", evidence_path.display()),
            "daemon redline_shadow_pm4 compared logits, KV, and recurrent-state hashes".into(),
        ],
    })
}

fn run_variant(config: &HookConfig, binary: &Path, label: &str) -> Result<VariantBench> {
    if !binary.is_file() {
        return Err(Error::Invalid(format!(
            "daemon binary does not exist: {}",
            binary.display()
        )));
    }
    let log = config.work_dir.join(format!("{label}.daemon.log"));
    let mut daemon = Daemon::start(binary, config, &log)?;
    let loaded = daemon.request(&json!({
        "type": "load",
        "model": config.model_path,
        "params": {
            "max_seq": config.max_seq,
            "kv_mode": config.kv_mode,
            "dflash_mode": "off"
        }
    }))?;
    if loaded.get("type").and_then(Value::as_str) != Some("loaded") {
        return Err(Error::Daemon(format!("unexpected load response: {loaded}")));
    }
    let capture = daemon.request(&json!({
        "type": "bench_decode",
        "context_tokens": config.decode_context,
        "iterations": 1,
        "redline_capture": true,
        "redline_detail": true
    }))?;
    for _ in 0..config.warmups {
        let response = shadow_request(&mut daemon, config)?;
        require_shadow_exact(&response)?;
    }
    let mut pm4_tok_s = Vec::with_capacity(config.samples);
    let mut hip_tok_s = Vec::with_capacity(config.samples);
    let mut snapshots = Vec::with_capacity(config.samples);
    let mut bit_exact = true;
    for _ in 0..config.samples {
        let response = shadow_request(&mut daemon, config)?;
        bit_exact &= require_shadow_exact(&response)?;
        let pm4_us = number(&response, "aql_host_us")?;
        let hip_us = number(&response, "hip_host_us")?;
        pm4_tok_s.push(config.decode_iterations as f64 * 1_000_000.0 / pm4_us);
        hip_tok_s.push(config.decode_iterations as f64 * 1_000_000.0 / hip_us);
        snapshots.push(response.get("aql").cloned().unwrap_or(Value::Null));
    }
    Ok(VariantBench {
        daemon: binary.to_owned(),
        pm4_tok_s,
        hip_tok_s,
        snapshots,
        bit_exact,
        capture,
    })
}

fn shadow_request(daemon: &mut Daemon, config: &HookConfig) -> Result<Value> {
    daemon.request(&json!({
        "type": "redline_shadow_pm4",
        "context_tokens": config.decode_context,
        "iterations": config.decode_iterations
    }))
}

fn require_shadow_exact(response: &Value) -> Result<bool> {
    let exact = response
        .get("bit_exact")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && response
            .get("blob_bit_exact")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    if !exact {
        return Err(Error::Daemon(
            "retained PM4 shadow failed logits/KV/recurrent parity".into(),
        ));
    }
    Ok(true)
}

fn compatible_snapshots(baseline: &VariantBench, candidate: &VariantBench) -> bool {
    baseline.bit_exact
        && candidate.bit_exact
        && !baseline.snapshots.is_empty()
        && baseline
            .snapshots
            .iter()
            .all(|value| Some(value) == baseline.snapshots.first())
        && candidate
            .snapshots
            .iter()
            .all(|value| Some(value) == candidate.snapshots.first())
        && baseline.snapshots.first() == candidate.snapshots.first()
}

fn candidate_daemon(config: &HookConfig, task: &CandidateTask) -> Result<PathBuf> {
    let from_plan = task
        .plan
        .as_ref()
        .and_then(|plan| plan.launch.get("hipfire"))
        .and_then(|value| value.get("candidate_daemon"))
        .and_then(Value::as_str)
        .or_else(|| {
            task.plan
                .as_ref()
                .and_then(|plan| plan.launch.get("candidate_daemon"))
                .and_then(Value::as_str)
        })
        .map(PathBuf::from);
    from_plan
        .or_else(|| config.candidate_daemon.clone())
        .ok_or_else(|| {
            Error::Invalid(
                "candidate daemon missing: set config candidate_daemon or plan.launch.hipfire.candidate_daemon"
                    .into(),
            )
        })
}

fn certify(config: &HookConfig, task: &CandidateTask) -> Result<Certification> {
    let candidate_daemon = candidate_daemon(config, task)?;
    let id = safe_component(&task.candidate.id);
    let baseline_shadow = run_variant(
        config,
        &config.baseline_daemon,
        &format!("{id}.cert-baseline-shadow"),
    )?;
    let candidate_shadow = run_variant(
        config,
        &candidate_daemon,
        &format!("{id}.cert-candidate-shadow"),
    )?;
    let shadow_passed = compatible_snapshots(&baseline_shadow, &candidate_shadow);
    let mut artifacts = BTreeMap::new();
    let shadow_path = evidence_path(config, &task.candidate.id, "cert-shadow");
    write_json(
        &shadow_path,
        &json!({"baseline": baseline_shadow, "candidate": candidate_shadow}),
    )?;
    artifacts.insert("shadow".into(), shadow_path);

    let mut notes = Vec::new();
    let product_ab_passed = if let Some(harness) = &config.serve_harness {
        let (passed, paths, harness_notes) =
            run_serve_abba(config, harness, task, &candidate_daemon)?;
        artifacts.extend(paths);
        notes.extend(harness_notes);
        passed
    } else {
        notes.push("serve_harness is not configured; product gate fails closed".into());
        false
    };
    Ok(Certification {
        schema_version: SCHEMA_VERSION,
        candidate_id: task.candidate.id.clone(),
        passed: shadow_passed && product_ab_passed,
        shadow_passed,
        product_ab_passed,
        artifacts,
        notes,
    })
}

fn run_serve_abba(
    config: &HookConfig,
    harness: &ServeHarnessConfig,
    task: &CandidateTask,
    candidate_daemon: &Path,
) -> Result<(bool, BTreeMap<String, PathBuf>, Vec<String>)> {
    let arms: [(&str, &Path); 4] = [
        ("a0", config.baseline_daemon.as_path()),
        ("b0", candidate_daemon),
        ("b1", candidate_daemon),
        ("a1", config.baseline_daemon.as_path()),
    ];
    let mut rows = BTreeMap::new();
    let mut artifacts = BTreeMap::new();
    for (index, (label, daemon)) in arms.into_iter().enumerate() {
        let output = config.work_dir.join(format!(
            "{}.serve-{label}.json",
            safe_component(&task.candidate.id)
        ));
        let mut command = Command::new(&harness.python);
        command
            .arg(&harness.script)
            .arg("--model")
            .arg(&config.model_path)
            .arg("--sampling")
            .arg(&harness.sampling)
            .arg("--kv")
            .arg(&config.kv_mode)
            .arg("--max-tokens")
            .arg(harness.max_tokens.to_string())
            .arg("--max-seq")
            .arg(config.max_seq.to_string())
            .arg("--mode")
            .arg(&harness.mode)
            .arg("--registry")
            .arg(&harness.registry)
            .arg("--out")
            .arg(&output)
            .arg("--port")
            .arg((harness.port + index as u16).to_string())
            .envs(&config.environment)
            .env("HIPFIRE_DAEMON_BIN", daemon)
            .env("HIP_VISIBLE_DEVICES", config.device_ordinal.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        if let Some(seed) = harness.seed {
            command.arg("--seed").arg(seed.to_string());
        }
        if let Some(session) = &harness.session {
            command.arg("--session").arg(session);
        }
        if let Some(prompts_file) = &harness.prompts_file {
            command.arg("--prompts-file").arg(prompts_file);
        }
        if let Some(tag) = &harness.tag {
            command.arg("--tag").arg(tag);
        }
        let status = command
            .spawn()?
            .wait_timeout(Duration::from_secs(harness.timeout_seconds))?;
        if !status.success() {
            return Err(Error::Command(format!(
                "serve_harness arm {label} exited with {status}"
            )));
        }
        let value: Value = serde_json::from_slice(&fs::read(&output)?)?;
        artifacts.insert(format!("serve_{label}"), output);
        rows.insert(label.to_owned(), value);
    }
    let a0 = rows.get("a0").expect("ABBA arm exists");
    let a1 = rows.get("a1").expect("ABBA arm exists");
    let b0 = rows.get("b0").expect("ABBA arm exists");
    let b1 = rows.get("b1").expect("ABBA arm exists");
    let content_exact = assistant_content(a0) == assistant_content(b0)
        && assistant_content(a1) == assistant_content(b1);
    let no_candidate_attractor = [b0, b1].into_iter().flat_map(rows_array).all(|row| {
        !row.get("attractor")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    });
    let candidate_nonempty = [b0, b1]
        .into_iter()
        .flat_map(rows_array)
        .all(|row| !row.get("empty").and_then(Value::as_bool).unwrap_or(true));
    let baseline_speed = [a0, a1]
        .into_iter()
        .flat_map(decode_speeds)
        .collect::<Vec<_>>();
    let candidate_speed = [b0, b1]
        .into_iter()
        .flat_map(decode_speeds)
        .collect::<Vec<_>>();
    let baseline_median = median(&baseline_speed)?;
    let candidate_median = median(&candidate_speed)?;
    let lift = lift_pct(baseline_median, candidate_median);
    let passed = content_exact
        && no_candidate_attractor
        && candidate_nonempty
        && lift >= config.min_lift_pct;
    let baseline_prefill = [a0, a1]
        .into_iter()
        .flat_map(prefill_speeds)
        .collect::<Vec<_>>();
    let candidate_prefill = [b0, b1]
        .into_iter()
        .flat_map(prefill_speeds)
        .collect::<Vec<_>>();
    let prefill_note = if baseline_prefill.is_empty() || candidate_prefill.is_empty() {
        "prefill=unreported".to_owned()
    } else {
        format!(
            "prefill baseline={:.3} tok/s candidate={:.3} tok/s",
            median(&baseline_prefill)?,
            median(&candidate_prefill)?
        )
    };
    Ok((
        passed,
        artifacts,
        vec![format!(
            "serve_harness ABBA: baseline={baseline_median:.3} tok/s candidate={candidate_median:.3} tok/s lift={lift:.3}% content_exact={content_exact} no_candidate_attractor={no_candidate_attractor} candidate_nonempty={candidate_nonempty}; {prefill_note}"
        )],
    ))
}

fn rows_array(value: &Value) -> impl Iterator<Item = &Value> {
    value.as_array().into_iter().flatten()
}

fn assistant_content(value: &Value) -> Vec<&str> {
    rows_array(value)
        .map(|row| {
            row.get("assistant_content")
                .and_then(Value::as_str)
                .unwrap_or("")
        })
        .collect()
}

fn decode_speeds(value: &Value) -> Vec<f64> {
    rows_array(value)
        .filter_map(|row| row.get("decode_tok_s").and_then(Value::as_f64))
        .collect()
}

fn prefill_speeds(value: &Value) -> Vec<f64> {
    rows_array(value)
        .filter_map(|row| row.get("prefill_tok_s").and_then(Value::as_f64))
        .collect()
}

fn number(value: &Value, key: &str) -> Result<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite() && *number > 0.0)
        .ok_or_else(|| Error::Daemon(format!("response has no positive {key}: {value}")))
}

fn median(values: &[f64]) -> Result<f64> {
    if values.is_empty() {
        return Err(Error::Invalid("cannot take median of empty samples".into()));
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Ok(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn lift_pct(baseline: f64, candidate: f64) -> f64 {
    if baseline <= 0.0 {
        0.0
    } else {
        100.0 * (candidate / baseline - 1.0)
    }
}

fn evidence_path(config: &HookConfig, candidate: &str, stage: &str) -> PathBuf {
    config
        .work_dir
        .join(format!("{}.{}.json", safe_component(candidate), stage))
}

fn safe_component(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))?;
    Ok(())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

trait WaitTimeout {
    fn wait_timeout(&mut self, timeout: Duration) -> Result<std::process::ExitStatus>;
}

impl WaitTimeout for Child {
    fn wait_timeout(&mut self, timeout: Duration) -> Result<std::process::ExitStatus> {
        let started = std::time::Instant::now();
        loop {
            if let Some(status) = self.try_wait()? {
                return Ok(status);
            }
            if started.elapsed() >= timeout {
                self.kill()?;
                let _ = self.wait();
                return Err(Error::Command(format!(
                    "command timed out after {} seconds",
                    timeout.as_secs()
                )));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_and_lift_are_stable() {
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
        assert!((lift_pct(100.0, 105.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn candidate_daemon_prefers_typed_launch() {
        let config: HookConfig = serde_json::from_value(json!({
            "schema_version": 1,
            "run": {"model":"m", "architecture":"gfx1201", "route":"decode", "baseline":"origin/redline"},
            "model_path": "model.hfq",
            "baseline_daemon": "base",
            "candidate_daemon": "fallback"
        }))
        .unwrap();
        let task: CandidateTask = serde_json::from_value(json!({
            "candidate": {"id": "c"},
            "plan": {"launch": {"hipfire": {"candidate_daemon": "candidate"}}}
        }))
        .unwrap();
        assert_eq!(
            candidate_daemon(&config, &task).unwrap(),
            PathBuf::from("candidate")
        );
    }

    #[test]
    fn snapshot_gate_checks_cross_binary_hashes() {
        let bench = |hash: &str| VariantBench {
            daemon: "daemon".into(),
            pm4_tok_s: vec![1.0],
            hip_tok_s: vec![1.0],
            snapshots: vec![json!({"logits_hash": hash, "kv_hash": "k", "recurrent_hash": "r"})],
            bit_exact: true,
            capture: Value::Null,
        };
        assert!(compatible_snapshots(&bench("a"), &bench("a")));
        assert!(!compatible_snapshots(&bench("a"), &bench("b")));
    }
}
