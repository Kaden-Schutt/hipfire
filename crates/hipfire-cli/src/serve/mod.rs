// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Serve host lifecycle and shared state.
//!
//! Concern: process/daemon spawning, admission control, model lifecycle, idle eviction,
//! and the `serve` subcommand entrypoints. Centralises all state that must be
//! shared across HTTP workers so transport changes stay isolated.

use crate::{
    config_bool, config_f64, config_i64, config_string, config_u64, find_daemon, find_model_path,
    http_get_json, list_local_models, load_params, probe_host, pull_command, resolved_for_model,
    resolved_global, ListArgs, Paths, PullArgs, ServeArgs, StopArgs,
};
use anyhow::{anyhow, bail, Context, Result};
use hipfire_client::Engine;
use hipfire_config::{load_catalog, load_global, resolve, ConfigLayer, ConfigSource, NamedLayer};
use hipfire_registry::{load as load_registry, RegistryV1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command},
    sync::{Arc, Condvar, Mutex},
    thread,
    time::{Duration, Instant},
};
use tiny_http::Server;

pub(crate) mod metrics;

pub mod complete;
pub mod http;
pub mod slots;

#[derive(Debug)]
pub(crate) struct ServeMeta {
    pub(crate) current_model: Option<String>,
    pub(crate) loading_model: Option<String>,
    pub(crate) instance_token: String,
    pub(crate) requests_served: u64,
    pub(crate) retries_attempted: u64,
    pub(crate) retries_succeeded: u64,
    pub(crate) recent_tok_s: Option<f64>,
    pub(crate) started: Instant,
    pub(crate) last_activity: Instant,
}

pub(crate) fn finish_prewarm(meta: &mut ServeMeta, succeeded: bool) {
    meta.loading_model = None;
    if succeeded {
        meta.last_activity = Instant::now();
    }
}

pub(crate) fn idle_model_expired(meta: &ServeMeta, idle_timeout: Duration) -> bool {
    meta.loading_model.is_none()
        && meta.current_model.is_some()
        && meta.last_activity.elapsed() >= idle_timeout
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ServePidRecord {
    pub(crate) pid: u32,
    #[serde(default)]
    pub(crate) start_time: Option<u64>,
    #[serde(default)]
    pub(crate) port: Option<u16>,
    #[serde(default)]
    pub(crate) token: Option<String>,
    #[serde(skip)]
    pub(crate) legacy: bool,
}

pub(crate) struct ServeRuntime {
    pub(crate) engine: Engine,
    pub(crate) paths: Paths,
    pub(crate) registry: RegistryV1,
    pub(crate) current_path: Option<PathBuf>,
    pub(crate) current_arch: Option<String>,
    pub(crate) continuous_batch_capable: bool,
    pub(crate) current_max_seq: u64,
    pub(crate) cache_capable: bool,
    pub(crate) kv_override: Option<String>,
    pub(crate) kv_backend_override: Option<String>,
    pub(crate) tp: Option<u64>,
    pub(crate) continuous_batch_size: u64,
}

pub(crate) struct ServeShared {
    pub(crate) runtime: Mutex<ServeRuntime>,
    pub(crate) meta: Mutex<ServeMeta>,
    pub(crate) max_request_bytes: u64,
    pub(crate) admission: Arc<Admission>,
    pub(crate) idle_timeout: Duration,
    pub(crate) retry_enabled: bool,
    pub(crate) retry_backoff: Duration,
    /// Test seam: when set, invoked instead of `thread::sleep` during retry backoff.
    pub(crate) backoff_hook: Mutex<Option<Arc<dyn Fn(Duration) + Send + Sync>>>,
    /// Concurrent multi-slot backend (`serve.multi_slot`). Deliberately NOT
    /// inside `runtime`: that mutex is exactly what serialises requests today,
    /// so an engine behind it would serve one caller at a time and change
    /// nothing.
    /// Prometheus counters and histograms for `/metrics`. Lock-free, so a
    /// scrape never contends with a request.
    pub(crate) metrics: metrics::Metrics,
    pub(crate) slot_engine: Option<Arc<slots::SlotBackend>>, // feature-gated type is still SlotBackend (stub when disabled)
}

#[derive(Debug, Default)]
pub(crate) struct AdmissionState {
    /// Batch-eligible requests currently admitted. The single-daemon backend
    /// caps this at one; the multi-slot engine admits `capacity` at once.
    eligible: usize,
    /// A batch-ineligible request holds the backend exclusively.
    ineligible_busy: bool,
    queued: usize,
    batch_model: Option<String>,
}

#[derive(Debug)]
pub(crate) struct Admission {
    state: Mutex<AdmissionState>,
    available: Condvar,
    max_queue: usize,
    timeout: Duration,
    /// How many batch-eligible requests may be in flight at once. One for the
    /// single-daemon backend -- this gate is what protects it -- and the slot
    /// count when the multi-slot engine is active, which has its own admission
    /// behind it.
    capacity: usize,
}

#[derive(Debug)]
pub(crate) struct AdmissionError {
    message: String,
    retry_after_seconds: u64,
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AdmissionError {}

#[derive(Debug)]
pub(crate) struct AdmissionGuard {
    admission: Arc<Admission>,
    is_eligible: bool,
    model: Option<String>,
}

impl Admission {
    pub(crate) fn new(max_queue: usize, timeout: Duration) -> Self {
        Self::new_with_capacity(max_queue, timeout, 1)
    }

    pub(crate) fn new_with_capacity(max_queue: usize, timeout: Duration, capacity: usize) -> Self {
        Self {
            state: Mutex::new(AdmissionState::default()),
            available: Condvar::new(),
            max_queue,
            timeout,
            capacity: capacity.max(1),
        }
    }

    pub(crate) fn is_model_compatible(current: &Option<String>, requested: Option<&str>) -> bool {
        match (current, requested) {
            (None, _) => true,
            (Some(cur), Some(req)) => cur == req,
            (Some(_), None) => true,
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn acquire(self: &Arc<Self>) -> std::result::Result<AdmissionGuard, AdmissionError> {
        self.acquire_for(false, None)
    }

    pub(crate) fn acquire_for(
        self: &Arc<Self>,
        is_eligible: bool,
        model: Option<&str>,
    ) -> std::result::Result<AdmissionGuard, AdmissionError> {
        let model_owned = model.map(|s| s.to_owned());
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        // Fast path when no queued waiters and resource is available.
        if is_eligible {
            if !state.ineligible_busy
                && state.eligible < self.capacity
                && state.queued == 0
                && Self::is_model_compatible(&state.batch_model, model_owned.as_deref())
            {
                state.eligible += 1;
                if state.batch_model.is_none() {
                    state.batch_model = model_owned.clone();
                }
                return Ok(AdmissionGuard {
                    admission: Arc::clone(self),
                    is_eligible: true,
                    model: model_owned,
                });
            }
        } else if state.eligible == 0 && !state.ineligible_busy && state.queued == 0 {
            state.ineligible_busy = true;
            return Ok(AdmissionGuard {
                admission: Arc::clone(self),
                is_eligible: false,
                model: None,
            });
        }
        if self.max_queue != 0 && state.queued >= self.max_queue {
            return Err(AdmissionError {
                message: format!(
                    "serve queue full (depth {}/{})",
                    state.queued, self.max_queue
                ),
                retry_after_seconds: self.retry_after_seconds(),
            });
        }
        state.queued = state.queued.saturating_add(1);
        let started = Instant::now();
        loop {
            if self.timeout.is_zero() {
                state = self
                    .available
                    .wait(state)
                    .unwrap_or_else(|error| error.into_inner());
            } else {
                let remaining = self.timeout.saturating_sub(started.elapsed());
                if remaining.is_zero() {
                    state.queued = state.queued.saturating_sub(1);
                    return Err(AdmissionError {
                        message: format!(
                            "serve queue wait exceeded {}ms",
                            self.timeout.as_millis()
                        ),
                        retry_after_seconds: self.retry_after_seconds(),
                    });
                }
                let (next, wait) = self
                    .available
                    .wait_timeout(state, remaining)
                    .unwrap_or_else(|error| error.into_inner());
                state = next;
                if wait.timed_out() {
                    let can_acquire = if is_eligible {
                        !state.ineligible_busy
                            && state.eligible < self.capacity
                            && Self::is_model_compatible(&state.batch_model, model_owned.as_deref())
                    } else {
                        state.eligible == 0 && !state.ineligible_busy
                    };
                    if !can_acquire {
                        state.queued = state.queued.saturating_sub(1);
                        return Err(AdmissionError {
                            message: format!(
                                "serve queue wait exceeded {}ms",
                                self.timeout.as_millis()
                            ),
                            retry_after_seconds: self.retry_after_seconds(),
                        });
                    }
                }
            }
            let can_acquire = if is_eligible {
                !state.ineligible_busy
                    && state.eligible < self.capacity
                    && Self::is_model_compatible(&state.batch_model, model_owned.as_deref())
            } else {
                state.eligible == 0 && !state.ineligible_busy
            };
            if can_acquire {
                state.queued = state.queued.saturating_sub(1);
                if is_eligible {
                    state.eligible += 1;
                    if state.batch_model.is_none() {
                        state.batch_model = model_owned.clone();
                    }
                    return Ok(AdmissionGuard {
                        admission: Arc::clone(self),
                        is_eligible: true,
                        model: model_owned.clone(),
                    });
                } else {
                    state.ineligible_busy = true;
                    return Ok(AdmissionGuard {
                        admission: Arc::clone(self),
                        is_eligible: false,
                        model: None,
                    });
                }
            }
        }
    }

    pub(crate) fn inflight(&self) -> usize {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.eligible + usize::from(state.ineligible_busy) + state.queued
    }

    pub(crate) fn retry_after_seconds(&self) -> u64 {
        if self.timeout.is_zero() {
            1
        } else {
            self.timeout.as_secs().max(1)
        }
    }
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        let mut state = self
            .admission
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if self.is_eligible {
            state.eligible = state.eligible.saturating_sub(1);
            if state.eligible == 0 {
                state.batch_model = None;
            }
        } else {
            state.ineligible_busy = false;
        }
        self.admission.available.notify_all();
    }
}

/// Conservative batch eligibility for independent continuous-batch decode.
///
/// Eligible only for Qwen (arch 5/6) or dense LFM2 (`lfm` arch identity),
/// stateless text with no tools/images/stops/spec/adaptive/prefix behavior.
/// TP policy: Qwen admits ordinary tp=1 or pure expert-parallel tp=4 when the
/// daemon advertises batch capability; dense LFM remains tp=1 only. All other
/// tp/arch combinations fall back to sequential. Check is intentionally strict
/// and synchronous; model arch is taken from `current_arch` when available,
/// otherwise inferred from the requested model name containing `qwen` or `lfm`.
///
/// Message-shape gate matches daemon admission: absent/empty `messages` are
/// eligible; otherwise only exactly one `user` message with plain string
/// content (no tool_calls, no multipart/array content).
pub(crate) fn is_batch_eligible_request(
    body: &serde_json::Value,
    tp: Option<u64>,
    current_arch: Option<&str>,
    daemon_batch_capable: bool,
) -> bool {
    if !daemon_batch_capable {
        return false;
    }
    // Qwen or LFM2 only. Prefer runtime arch when known.
    let (is_qwen, is_lfm) = if let Some(arch) = current_arch {
        let arch_l = arch.to_ascii_lowercase();
        (arch_l.contains("qwen"), arch_l.contains("lfm"))
    } else if let Some(model) = body.get("model").and_then(|v| v.as_str()) {
        let model_l = model.to_ascii_lowercase();
        (model_l.contains("qwen"), model_l.contains("lfm"))
    } else {
        (false, false)
    };
    if !is_qwen && !is_lfm {
        return false;
    }
    // TP policy: Qwen tp=1 ordinary or tp=4 pure EP; dense LFM tp=1 only.
    let tp_degree = tp.unwrap_or(1);
    let tp_ok = if is_qwen {
        tp_degree == 1 || tp_degree == 4
    } else {
        tp_degree == 1
    };
    if !tp_ok {
        return false;
    }
    // Stateless text: no tools, no images, no stops, no spec, no adaptive, no prefix.
    if body
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty())
    {
        return false;
    }
    if body.get("tool_choice").is_some() {
        return false;
    }
    // Images via explicit image_base64.
    if body.get("image_base64").is_some() {
        return false;
    }
    // Message history must match daemon single-user plain-string shape.
    if !batch_messages_are_single_user(body) {
        return false;
    }
    if body.get("stop").is_some() {
        return false;
    }
    // Speculation / adaptive / prefix behavior disqualifies.
    for key in [
        "speculation",
        "dflash_mode",
        "mtp_mode",
        "ngram_draft",
        "prefill_sparse_threshold",
        "kv_adaptive",
        "prefix",
    ] {
        if body.get(key).is_some() {
            return false;
        }
    }
    true
}

/// HTTP/OpenAI `messages` are batch-eligible only when absent/empty (prompt
/// path) or exactly one user turn with plain string content. Multi-turn,
/// system/assistant/tool roles, tool_call payloads, and multipart/image
/// content stay on the sequential route — same contract as the daemon.
pub(crate) fn batch_messages_are_single_user(body: &serde_json::Value) -> bool {
    let Some(messages) = body.get("messages") else {
        return true;
    };
    let Some(arr) = messages.as_array() else {
        return false;
    };
    if arr.is_empty() {
        return true;
    }
    if arr.len() != 1 {
        return false;
    }
    let m0 = &arr[0];
    if m0.get("role").and_then(|v| v.as_str()) != Some("user") {
        return false;
    }
    // Tool-call payloads on the sole message force sequential (batch v1 has
    // no tools path). Empty / missing tool_calls is fine.
    if let Some(tc) = m0.get("tool_calls") {
        if tc.as_array().is_some_and(|a| !a.is_empty()) || tc.is_object() {
            return false;
        }
    }
    // Content must be a plain string (reject multipart/array/image parts).
    match m0.get("content") {
        Some(serde_json::Value::String(_)) => true,
        // Missing content is not a plain user string turn.
        None => false,
        // Arrays (multipart/text+image), objects, numbers, bool, null.
        Some(_) => false,
    }
}

pub(crate) fn serve_command(paths: &Paths, mut args: ServeArgs) -> Result<()> {
    let (_, resolved) = resolved_global(paths, true)?;
    let default_host = config_string(&resolved, "serve.host")?;
    let default_port = config_u64(&resolved, "serve.port")? as u16;
    let (host, port, positional_model) =
        resolve_serve_positionals(paths, &args.positionals, &default_host, default_port)?;
    if let Some(positional_model) = positional_model {
        if args
            .model
            .as_ref()
            .is_some_and(|model| model != &positional_model)
        {
            bail!("serve model specified more than once");
        }
        args.model = Some(positional_model);
    }
    if args.detach && !args.foreground_child {
        return detach_serve(paths, &args, &host, port);
    }
    serve_foreground(paths, &args, &host, port, resolved)
}

pub(crate) fn resolve_serve_positionals(
    paths: &Paths,
    values: &[String],
    default_host: &str,
    default_port: u16,
) -> Result<(String, u16, Option<String>)> {
    let registry = load_registry(&paths.registry).registry;
    let mut host = None;
    let mut port = None;
    let mut model = None;
    for value in values {
        if let Ok(value_port) = value.parse::<u16>() {
            if port.replace(value_port).is_some() {
                bail!("serve port specified more than once");
            }
            continue;
        }
        if let Some((value_host, value_port)) = parse_host_port(value)? {
            if host.replace(value_host).is_some() || port.replace(value_port).is_some() {
                bail!("serve bind specified more than once");
            }
            continue;
        }
        let is_model =
            registry.model(value).is_some() || find_model_path(paths, &registry, value).is_some();
        if is_model && model.is_none() {
            model = Some(value.clone());
        } else if host.replace(value.clone()).is_some() {
            bail!("serve host specified more than once");
        }
    }
    Ok((
        host.unwrap_or_else(|| default_host.to_owned()),
        port.unwrap_or(default_port),
        model,
    ))
}

pub(crate) fn parse_host_port(value: &str) -> Result<Option<(String, u16)>> {
    if let Some(stripped) = value.strip_prefix('[') {
        if let Some((host, port)) = stripped.split_once("]:") {
            return Ok(Some((
                host.to_owned(),
                port.parse().context("invalid serve port")?,
            )));
        }
    }
    if value.matches(':').count() == 1 {
        if let Some((host, port)) = value.rsplit_once(':') {
            if let Ok(port) = port.parse::<u16>() {
                return Ok(Some((host.to_owned(), port)));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
pub(crate) fn parse_bind(
    address: Option<&str>,
    port: Option<u16>,
    default_host: &str,
    default_port: u16,
) -> Result<(String, u16)> {
    let Some(address) = address else {
        return Ok((default_host.to_owned(), port.unwrap_or(default_port)));
    };
    if let Ok(port_only) = address.parse::<u16>() {
        return Ok((default_host.to_owned(), port_only));
    }
    if let Some(stripped) = address.strip_prefix('[') {
        if let Some((host, port_text)) = stripped.split_once("]:") {
            return Ok((
                host.to_owned(),
                port_text.parse().context("invalid serve port")?,
            ));
        }
    }
    if address.matches(':').count() == 1 {
        if let Some((host, port_text)) = address.rsplit_once(':') {
            if let Ok(parsed) = port_text.parse::<u16>() {
                return Ok((host.to_owned(), parsed));
            }
        }
    }
    Ok((address.to_owned(), port.unwrap_or(default_port)))
}

pub(crate) fn detach_serve(paths: &Paths, args: &ServeArgs, host: &str, port: u16) -> Result<()> {
    fs::create_dir_all(&paths.root)?;
    let log_path = paths.root.join("serve.log");
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))?;
    let executable = env::current_exe().context("failed to resolve native hipfire binary")?;
    let mut command = Command::new(executable);
    command
        .arg("serve")
        .arg(host)
        .arg(port.to_string())
        .arg("--foreground-child")
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    if args.no_prewarm {
        command.arg("--no-prewarm");
    }
    if let Some(model) = &args.model {
        command.arg("--model").arg(model);
    }
    if let Some(mode) = &args.kv_mode {
        command.arg("--kv-mode").arg(mode);
    }
    if let Some(backend) = &args.kv_backend {
        command.arg("--kv-backend").arg(backend);
    }
    if let Some(seconds) = args.idle_timeout {
        command.arg("--idle-timeout").arg(seconds.to_string());
    }
    if let Some(tp) = args.tp {
        command.arg("--tp").arg(tp.to_string());
    }
    if let Some(batch) = args.continuous_batch_size {
        command
            .arg("--continuous-batch-size")
            .arg(batch.to_string());
    }
    let mut child = command.spawn().context("failed to detach native serve")?;
    let probe_host = match host {
        "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        other => other,
    };
    for _ in 0..600 {
        if let Some(status) = child.try_wait()? {
            bail!(
                "native serve exited before readiness ({status}); see {}",
                log_path.display()
            );
        }
        if health_ready(probe_host, port) {
            println!(
                "hipfire serve running at http://{}:{} (PID {}, log {})",
                host,
                port,
                child.id(),
                log_path.display()
            );
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    bail!(
        "native serve did not become ready within 60s; PID {}, see {}",
        child.id(),
        log_path.display()
    )
}

pub(crate) fn health_ready(host: &str, port: u16) -> bool {
    let url = if host.contains(':') {
        format!("http://[{host}]:{port}/health")
    } else {
        format!("http://{host}:{port}/health")
    };
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(100)))
        .http_status_as_error(false)
        .build()
        .into();
    agent
        .get(&url)
        .call()
        .is_ok_and(|response| response.status().is_success())
}

pub(crate) fn serve_foreground(
    paths: &Paths,
    args: &ServeArgs,
    host: &str,
    port: u16,
    global: hipfire_config::ResolvedConfig,
) -> Result<()> {
    let daemon = find_daemon(paths).ok_or_else(|| anyhow!("daemon binary not found"))?;
    let registry = load_registry(&paths.registry).registry;
    let process_config = hipfire_config::ProcessConfig::from_resolved(&global)?;
    let mut engine = Engine::spawn_configured(&daemon, &BTreeMap::new(), &process_config)?;
    engine.ping()?;
    let max_request_bytes = config_u64(&global, "serve.max_request_bytes")?;
    let max_queue = config_u64(&global, "serve.max_queue")? as usize;
    let queue_timeout = Duration::from_millis(config_u64(&global, "serve.queue_timeout_ms")?);
    let continuous_batch_size = args
        .continuous_batch_size
        .unwrap_or(config_u64(&global, "serve.continuous_batch_size")?);
    if continuous_batch_size == 0 || continuous_batch_size > 256 {
        bail!("--continuous-batch-size must be between 1 and 256");
    }
    let retry_enabled = config_bool(&global, "serve.retry_enabled")?;
    let retry_backoff = Duration::from_millis(config_u64(&global, "serve.retry_backoff_ms")?);
    let idle_timeout = Duration::from_secs(
        args.idle_timeout
            .unwrap_or(config_u64(&global, "serve.idle_timeout_seconds")?),
    );
    let default_model = args
        .model
        .clone()
        .unwrap_or(config_string(&global, "serve.default_model")?);
    let instance_token = serve_instance_token();
    // Opt-in concurrent backend. Built before ServeShared so a failure here is
    // a clean startup error rather than a half-configured server.
    let slot_engine: Option<Arc<slots::SlotBackend>> =
        match config_bool(&global, "serve.multi_slot") {
            Ok(true) => {
                let model_path = find_model_path(paths, &registry, &default_model)
                    .ok_or_else(|| anyhow!("multi_slot: cannot resolve model {default_model}"))?;
                let n_slots = config_u64(&global, "serve.multi_slot_slots").unwrap_or(4) as usize;
                let cap_tokens =
                    config_u64(&global, "serve.multi_slot_ctx").unwrap_or(8192) as usize;
                let mut hfq = hipfire_runtime::hfq::HfqFile::open(&model_path)
                    .with_context(|| format!("multi_slot: open {}", model_path.display()))?;
                let tokenizer =
                    hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                        .map_err(|e| anyhow!("multi_slot: tokenizer: {e}"))?;
                drop(hfq);
                // The engine runs IN this process, so hardware.devices /
                // HIPFIRE_DEVICES must be applied here — the daemon does this in
                // its own startup, which an in-process backend never runs.
                // Without it the engine always lands on device 0.
                hipfire_config::apply_device_visibility(&process_config)
                    .map_err(|e| anyhow!("multi_slot: device visibility: {e}"))?;
                let engine = hipfire_arch_qwen35::serve_engine::SlotEngine::spawn(
                    hipfire_arch_qwen35::serve_engine::EngineConfig {
                        model_path: model_path.clone(),
                        n_slots,
                        cap_tokens,
                        prefill_chunk: config_u64(&global, "serve.multi_slot_prefill_chunk")
                            .unwrap_or(1024) as usize,
                        host_budget_bytes: 16 * 1024 * 1024 * 1024,
                        swap_dir: std::env::temp_dir().join("hipfire-serve-swap"),
                    },
                )
                .map_err(|e| anyhow!("multi_slot: {e}"))?;
                eprintln!(
                    "serve: multi-slot backend up ({} slots, {} ctx) - requests run concurrently",
                    n_slots, cap_tokens
                );
                Some(Arc::new(slots::SlotBackend { engine, tokenizer }))
            }
            _ => None,
        };
    #[cfg(feature = "multi-slot")]
    let slot_engine_active = slot_engine.is_some();
    #[cfg(not(feature = "multi-slot"))]
    let slot_engine: Option<Arc<slots::SlotBackend>> = None;
    #[cfg(not(feature = "multi-slot"))]
    let slot_engine_active = false;
    let slot_concurrency = if slot_engine_active {
        config_u64(&global, "serve.multi_slot_slots").unwrap_or(4) as usize
    } else {
        1
    };
    let shared = Arc::new(ServeShared {
        metrics: metrics::Metrics::default(),
        slot_engine,
        runtime: Mutex::new(ServeRuntime {
            engine,
            paths: paths.clone(),
            registry: registry.clone(),
            current_path: None,
            current_arch: None,
            continuous_batch_capable: false,
            current_max_seq: 0,
            cache_capable: false,
            kv_override: args.kv_mode.clone(),
            kv_backend_override: args.kv_backend.clone(),
            tp: args.tp,
            continuous_batch_size,
        }),
        meta: Mutex::new(ServeMeta {
            current_model: None,
            loading_model: None,
            instance_token: instance_token.clone(),
            requests_served: 0,
            retries_attempted: 0,
            retries_succeeded: 0,
            recent_tok_s: None,
            started: Instant::now(),
            last_activity: Instant::now(),
        }),
        max_request_bytes,
        // The HTTP gate must admit as many requests as the live backend can
        // actually run, or they queue here and never reach it -- the exact
        // serialisation both batching paths exist to remove. Two independent
        // mechanisms feed this: the multi-slot engine's slot count (1 when it
        // is off) and the single-daemon continuous batch width. Take the
        // larger; each has its own admission behind this gate.
        admission: Arc::new(Admission::new_with_capacity(
            max_queue,
            queue_timeout,
            slot_concurrency.max(continuous_batch_size as usize),
        )),
        idle_timeout,
        retry_enabled,
        retry_backoff,
        backoff_hook: Mutex::new(None),
    });
    let bind = format_bind(host, port);
    let server = Server::http(&bind).map_err(|error| anyhow!("failed to bind {bind}: {error}"))?;
    fs::create_dir_all(&paths.root)?;
    let pid_path = paths.root.join("serve.pid");
    let pid_record = ServePidRecord {
        pid: std::process::id(),
        start_time: proc_start_time(std::process::id()),
        port: Some(port),
        token: Some(instance_token),
        legacy: false,
    };
    fs::write(
        &pid_path,
        format!("{}\n", serde_json::to_string(&pid_record)?),
    )?;
    let cleanup = pid_path.clone();
    ctrlc::set_handler(move || {
        let _ = fs::remove_file(&cleanup);
        std::process::exit(0);
    })
    .context("failed to install serve signal handler")?;
    eprintln!("[hipfire] native serve listening on http://{bind}");
    if !args.no_prewarm && !slot_engine_active {
        let shared = Arc::clone(&shared);
        thread::spawn(move || {
            shared
                .meta
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .loading_model = Some(default_model.clone());
            let result = shared
                .runtime
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .ensure_model(&default_model, &shared.meta, None);
            {
                let mut meta = shared
                    .meta
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                finish_prewarm(&mut meta, result.is_ok());
            }
            match result {
                Ok(_) => eprintln!("[hipfire] pre-warmed {default_model}"),
                Err(error) => eprintln!("[hipfire] pre-warm failed: {error:#}; serving lazily"),
            }
        });
    } else if !args.no_prewarm {
        // The multi-slot engine already holds trunk + vision weights; the
        // legacy runtime path would load a second full copy and OOM consumer
        // cards. All completion requests route to the slot backend when it is
        // up, so eager pre-warm here is pure waste.
        // /health reports meta.current_model; the warm gate (and any client
        // readiness probe) needs it to name the served model even though the
        // legacy runtime never loaded anything.
        shared
            .meta
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .current_model = Some(default_model.clone());
        eprintln!("[hipfire] multi-slot backend owns the model; skipping legacy pre-warm");
    }
    if !shared.idle_timeout.is_zero() {
        let shared = Arc::clone(&shared);
        thread::spawn(move || loop {
            thread::sleep(Duration::from_secs(1));
            if shared.admission.inflight() != 0 {
                continue;
            }
            let expired = {
                let meta = shared
                    .meta
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                idle_model_expired(&meta, shared.idle_timeout)
            };
            if !expired {
                continue;
            }
            let unloaded = {
                let mut runtime = shared
                    .runtime
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                if runtime.current_path.is_some() {
                    let result = runtime.engine.unload();
                    if result.is_ok() {
                        runtime.current_path = None;
                        runtime.current_arch = None;
                        runtime.current_max_seq = 0;
                        runtime.cache_capable = false;
                    }
                    result
                } else {
                    Ok(())
                }
            };
            if unloaded.is_ok() {
                let mut meta = shared
                    .meta
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                meta.current_model = None;
                meta.loading_model = None;
                meta.last_activity = Instant::now();
                eprintln!("[hipfire] unloaded idle model");
            }
        });
    }
    for request in server.incoming_requests() {
        let shared = Arc::clone(&shared);
        thread::spawn(move || {
            if let Err(error) = crate::serve::http::handle_http(request, shared) {
                eprintln!("[hipfire] HTTP request failed: {error:#}");
            }
        });
    }
    let _ = fs::remove_file(pid_path);
    Ok(())
}

pub(crate) fn format_bind(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

pub(crate) fn should_prewarm_qwen_mq4r_decode(
    path: &Path,
    loaded: &serde_json::Value,
    tp: Option<u64>,
) -> bool {
    let qwen = loaded
        .get("arch")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|arch| arch.starts_with("qwen3_5"));
    let mq4r = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mq4r"));
    qwen && mq4r && tp.unwrap_or(1) == 1
}

pub(crate) fn prewarm_qwen_mq4r_decode(engine: &mut Engine) -> Result<()> {
    let response = engine.request(&serde_json::json!({
        "type": "bench_decode",
        "context_tokens": 64,
        "iterations": 32,
    }))?;
    match response.get("type").and_then(serde_json::Value::as_str) {
        Some("decode_result") => {
            let tok_s = response
                .get("tok_s")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            eprintln!("[hipfire] pre-warmed Qwen MQ4R decode route ({tok_s:.1} tok/s)");
            Ok(())
        }
        Some("error") => bail!(
            "Qwen MQ4R decode pre-warm failed: {}",
            response
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("daemon returned an unspecified error")
        ),
        other => bail!(
            "Qwen MQ4R decode pre-warm expected decode_result, received {}",
            other.unwrap_or("missing type")
        ),
    }
}

impl ServeRuntime {
    pub(crate) fn ensure_model(
        &mut self,
        model: &str,
        meta: &Mutex<ServeMeta>,
        minimum_max_seq: Option<u64>,
    ) -> Result<hipfire_config::ResolvedConfig> {
        let (tag, entry) = self
            .registry
            .model(model)
            .map(|(tag, entry)| (Some(tag.to_owned()), Some(entry)))
            .unwrap_or((None, None));
        let mut path = find_model_path(&self.paths, &self.registry, model);
        if path.is_none() && entry.is_some() {
            pull_command(
                &self.paths,
                PullArgs {
                    model: model.to_owned(),
                    force: false,
                },
            )?;
            path = entry.map(|entry| self.paths.models.join(&entry.file));
        }
        let path = path.ok_or_else(|| anyhow!("model not found locally: {model}"))?;
        let resolved = resolved_for_model(&self.paths, model, tag.as_deref(), entry)?;
        let must_reload = self.current_path.as_ref() != Some(&path)
            || minimum_max_seq.is_some_and(|minimum| self.current_max_seq < minimum);
        if must_reload {
            let max_tokens = minimum_max_seq
                .map(|minimum| minimum.saturating_sub(1024))
                .unwrap_or(config_u64(&resolved, "generation.max_tokens")?);
            let mut params = load_params(
                &resolved,
                entry,
                &path,
                max_tokens,
                self.kv_override.as_deref(),
                self.kv_backend_override.as_deref(),
            )?;
            if let Some(tp) = self.tp {
                params["tp"] = serde_json::json!(tp);
            }
            params["continuous_batch_size"] = serde_json::json!(self.continuous_batch_size);
            let loaded_max_seq = params["max_seq"].as_u64().unwrap_or(0);
            if minimum_max_seq.is_some() {
                eprintln!("[hipfire] bumping load max_seq to {loaded_max_seq} for request budget");
            }
            let loaded = self.engine.load(&path, params)?;
            if should_prewarm_qwen_mq4r_decode(&path, &loaded, self.tp) {
                prewarm_qwen_mq4r_decode(&mut self.engine)?;
            }
            self.cache_capable = loaded
                .get("cache_capable")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            self.current_path = Some(path);
            self.current_arch = loaded
                .get("arch")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            self.continuous_batch_capable = loaded
                .get("continuous_batch_capable")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            self.current_max_seq = loaded_max_seq;
            meta.lock()
                .unwrap_or_else(|error| error.into_inner())
                .current_model = Some(tag.unwrap_or_else(|| model.to_owned()));
        }
        Ok(resolved)
    }
}

pub(crate) fn serve_instance_token() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut digest = Sha256::new();
    digest.update(std::process::id().to_le_bytes());
    digest.update(now.to_le_bytes());
    format!("{:x}", digest.finalize())
}

pub(crate) fn proc_start_time(pid: u32) -> Option<u64> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_comm = stat.rsplit_once(") ")?.1;
    after_comm.split_whitespace().nth(19)?.parse().ok()
}

pub(crate) fn pid_owns_listen_port(pid: u32, port: u16) -> Option<bool> {
    let mut listen_inodes = BTreeSet::new();
    let port_hex = format!("{port:04X}");
    let mut read_any = false;
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let Ok(raw) = fs::read_to_string(table) else {
            continue;
        };
        read_any = true;
        for line in raw.lines().skip(1) {
            let columns = line.split_whitespace().collect::<Vec<_>>();
            if columns.len() < 10 || columns[3] != "0A" {
                continue;
            }
            let Some((_, local_port)) = columns[1].rsplit_once(':') else {
                continue;
            };
            if local_port.eq_ignore_ascii_case(&port_hex) {
                listen_inodes.insert(columns[9].to_owned());
            }
        }
    }
    if !read_any {
        return None;
    }
    if listen_inodes.is_empty() {
        return Some(false);
    }
    let entries = fs::read_dir(format!("/proc/{pid}/fd")).ok()?;
    for entry in entries.flatten() {
        let Ok(target) = fs::read_link(entry.path()) else {
            continue;
        };
        let target = target.to_string_lossy();
        if let Some(inode) = target
            .strip_prefix("socket:[")
            .and_then(|value| value.strip_suffix(']'))
        {
            if listen_inodes.contains(inode) {
                return Some(true);
            }
        }
    }
    Some(false)
}

pub(crate) fn validate_serve_pid(
    record: &ServePidRecord,
    host: &str,
    fallback_port: u16,
) -> Result<()> {
    let proc_dir = PathBuf::from(format!("/proc/{}", record.pid));
    if !proc_dir.is_dir() {
        bail!("tracked serve PID {} is no longer alive", record.pid);
    }
    let cmdline = fs::read(proc_dir.join("cmdline")).unwrap_or_default();
    let cmdline = String::from_utf8_lossy(&cmdline).replace('\0', " ");
    if !cmdline.contains("hipfire") || !cmdline.contains("serve") {
        bail!("PID {} is not a hipfire serve process", record.pid);
    }
    if let Some(expected) = record.start_time {
        if proc_start_time(record.pid) != Some(expected) {
            bail!("PID {} was reused after serve.pid was written", record.pid);
        }
    }

    let port = record.port.unwrap_or(fallback_port);
    let owns_port = pid_owns_listen_port(record.pid, port);
    if owns_port == Some(false) {
        bail!(
            "PID {} does not own the tracked serve port {port}",
            record.pid
        );
    }
    let health_matches = record.token.as_deref().is_some_and(|expected| {
        http_get_json(host, port, "/health").is_some_and(|health| {
            health.get("pid").and_then(serde_json::Value::as_u64) == Some(record.pid as u64)
                && health.get("token").and_then(serde_json::Value::as_str) == Some(expected)
        })
    });
    if owns_port == Some(true) || health_matches || record.legacy && owns_port.is_none() {
        Ok(())
    } else {
        bail!(
            "could not prove ownership of PID {} with port or health token",
            record.pid
        )
    }
}

pub(crate) fn stop_command(paths: &Paths, args: StopArgs) -> Result<()> {
    let pid_path = paths.root.join("serve.pid");
    match fs::read_to_string(&pid_path) {
        Ok(raw) => {
            let record = parse_pid_record(&raw)
                .ok_or_else(|| anyhow!("invalid serve.pid; refusing to signal"))?;
            let resolved = resolved_global(paths, true)
                .ok()
                .map(|(_, resolved)| resolved);
            let host = resolved
                .as_ref()
                .and_then(|resolved| config_string(resolved, "serve.host").ok())
                .unwrap_or_else(|| "127.0.0.1".into());
            let fallback_port = args
                .port
                .or_else(|| {
                    resolved.as_ref().and_then(|resolved| {
                        config_u64(resolved, "serve.port")
                            .ok()
                            .and_then(|port| u16::try_from(port).ok())
                    })
                })
                .unwrap_or(11435);
            if let Err(error) = validate_serve_pid(&record, probe_host(&host), fallback_port) {
                fs::remove_file(&pid_path)?;
                if !args.force {
                    bail!("{error}; removed stale pidfile without signaling");
                }
                eprintln!(
                    "warning: {error}; refusing direct PID signal and continuing forced reap"
                );
            } else {
                let status = Command::new("kill")
                    .arg("-TERM")
                    .arg(record.pid.to_string())
                    .status()
                    .context("failed to invoke kill")?;
                if !status.success() {
                    bail!("failed to stop native serve PID {}", record.pid);
                }
                for _ in 0..50 {
                    if !Path::new(&format!("/proc/{}", record.pid)).exists() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                let _ = fs::remove_file(&pid_path);
                println!("hipfire serve stopped (PID {})", record.pid);
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            println!("hipfire serve is not running");
        }
        Err(error) => return Err(error).context("failed to read serve.pid"),
    }
    if args.force || args.all {
        let (_, resolved) = resolved_global(paths, true)?;
        let port = args
            .port
            .unwrap_or(config_u64(&resolved, "serve.port")? as u16);
        let _ = Command::new("pkill").args(["-x", "daemon"]).status();
        if args.all {
            let _ = Command::new("pkill")
                .args(["-f", "target/release/hipfire-quantize"])
                .status();
        }
        let _ = Command::new("fuser")
            .args(["-k", &format!("{port}/tcp")])
            .status();
        println!("reaped orphan daemon processes and freed port {port}");
    }
    Ok(())
}

pub(crate) fn parse_pid_record(raw: &str) -> Option<ServePidRecord> {
    if let Ok(pid) = raw.trim().parse() {
        return Some(ServePidRecord {
            pid,
            start_time: None,
            port: None,
            token: None,
            legacy: true,
        });
    }
    let mut record = serde_json::from_str::<ServePidRecord>(raw).ok()?;
    record.legacy = record.start_time.is_none() && record.port.is_none() && record.token.is_none();
    Some(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Commands, Paths, ServeArgs, StopArgs};
    use clap::Parser;
    use hipfire_config::{
        load_global, resolve, ConfigLayer, ConfigPaths, ConfigSource, NamedLayer,
    };
    use hipfire_registry::{RegistryPaths, RegistryV1};
    use std::{
        env, fs,
        path::{Path, PathBuf},
        sync::{mpsc, Arc, Mutex},
        thread,
        time::{Duration, Instant},
    };
    fn test_paths(label: &str) -> Paths {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hipfire-cli-{label}-{}-{nonce}",
            std::process::id()
        ));
        let config = ConfigPaths::under(&root);
        Paths {
            models: config.models.clone(),
            registry: RegistryPaths {
                cache: root.join("registry.cache.json"),
            },
            root,
            config,
        }
    }

    fn idle_test_meta() -> ServeMeta {
        ServeMeta {
            current_model: Some("model.hfq".to_owned()),
            loading_model: Some("model.hfq".to_owned()),
            instance_token: "test".to_owned(),
            requests_served: 0,
            retries_attempted: 0,
            retries_succeeded: 0,
            recent_tok_s: None,
            started: Instant::now(),
            last_activity: Instant::now() - Duration::from_secs(600),
        }
    }

    #[test]
    fn idle_timeout_does_not_evict_a_loading_model() {
        let meta = idle_test_meta();
        assert!(!idle_model_expired(&meta, Duration::from_secs(300)));
    }

    #[test]
    fn successful_prewarm_starts_a_fresh_idle_window() {
        let mut meta = idle_test_meta();
        finish_prewarm(&mut meta, true);
        assert!(meta.loading_model.is_none());
        assert!(!idle_model_expired(&meta, Duration::from_secs(300)));
        assert!(meta.last_activity.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn bind_and_pid_compatibility_parsers_cover_legacy_shapes() {
        assert_eq!(
            parse_bind(Some("127.0.0.1:12000"), None, "0.0.0.0", 11435).unwrap(),
            ("127.0.0.1".into(), 12000)
        );
        assert_eq!(
            parse_bind(Some("[::1]:12001"), None, "0.0.0.0", 11435).unwrap(),
            ("::1".into(), 12001)
        );
        let legacy = parse_pid_record("42\n").unwrap();
        assert_eq!(legacy.pid, 42);
        assert!(legacy.legacy);
        let json = parse_pid_record(r#"{"pid":43,"token":"old"}"#).unwrap();
        assert_eq!(json.pid, 43);
        assert_eq!(json.token.as_deref(), Some("old"));
        assert!(!json.legacy);
    }

    #[test]
    fn serve_accepts_legacy_positionals_and_native_overrides() {
        let parsed = Cli::try_parse_from([
            "hipfire",
            "serve",
            "qwen3.6:35b-a3b-mq4r",
            "127.0.0.1",
            "11520",
            "--kv-mode",
            "q8",
            "--kv-backend",
            "vmm",
            "--idle-timeout",
            "0",
            "--tp",
            "2",
        ])
        .unwrap();
        let Some(Commands::Serve(args)) = parsed.command else {
            panic!("expected serve command")
        };
        assert_eq!(args.positionals.len(), 3);
        assert_eq!(args.kv_mode.as_deref(), Some("q8"));
        assert_eq!(args.kv_backend.as_deref(), Some("vmm"));
        assert_eq!(args.idle_timeout, Some(0));
        assert_eq!(args.tp, Some(2));
    }

    #[test]
    fn admission_queue_is_bounded_and_times_out() {
        let admission = Arc::new(Admission::new(1, Duration::from_millis(200)));
        let holder = admission.acquire().unwrap();
        let queued_admission = Arc::clone(&admission);
        let (sender, receiver) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let guard = queued_admission.acquire().unwrap();
            sender.send(()).unwrap();
            drop(guard);
        });
        for _ in 0..100 {
            if admission.inflight() == 2 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        let saturated = admission.acquire().unwrap_err();
        assert!(saturated.message.contains("queue full"));
        drop(holder);
        receiver.recv_timeout(Duration::from_secs(1)).unwrap();
        waiter.join().unwrap();

        let admission = Arc::new(Admission::new(1, Duration::from_millis(5)));
        let _holder = admission.acquire().unwrap();
        let timeout = admission.acquire().unwrap_err();
        assert!(timeout.message.contains("wait exceeded"));
        assert_eq!(admission.inflight(), 1);
    }

    #[test]
    fn admission_eligible_concurrent_up_to_capacity() {
        let admission = Arc::new(Admission::new_with_capacity(8, Duration::from_secs(1), 2));
        let g1 = admission.acquire_for(true, Some("qwen3.5:7b")).unwrap();
        let g2 = admission.acquire_for(true, Some("qwen3.5:7b")).unwrap();
        assert_eq!(admission.inflight(), 2);
        // Third eligible same model should queue, then timeout quickly.
        let admission2 = Arc::clone(&admission);
        let handle =
            thread::spawn(move || admission2.acquire_for(true, Some("qwen3.5:7b")).unwrap());
        thread::sleep(Duration::from_millis(50));
        assert_eq!(admission.inflight(), 3); // 2 held + 1 queued
        drop(g1);
        let g3 = handle.join().unwrap();
        assert_eq!(admission.inflight(), 2);
        drop(g2);
        drop(g3);
        assert_eq!(admission.inflight(), 0);
    }

    #[test]
    fn admission_ineligible_is_exclusive() {
        let admission = Arc::new(Admission::new_with_capacity(8, Duration::from_secs(1), 2));
        let g1 = admission.acquire_for(true, Some("qwen3.5:7b")).unwrap();
        // Ineligible must wait for eligible to finish, even though capacity not full.
        let admission2 = Arc::clone(&admission);
        let handle = thread::spawn(move || admission2.acquire().unwrap());
        thread::sleep(Duration::from_millis(50));
        assert!(admission.inflight() == 2); // 1 eligible + 1 queued ineligible
        drop(g1);
        let g_inelig = handle.join().unwrap();
        assert_eq!(admission.inflight(), 1);
        // While ineligible holds, eligible must wait.
        let admission3 = Arc::clone(&admission);
        let handle2 =
            thread::spawn(move || admission3.acquire_for(true, Some("qwen3.5:7b")).unwrap());
        thread::sleep(Duration::from_millis(50));
        assert_eq!(admission.inflight(), 2);
        drop(g_inelig);
        let g2 = handle2.join().unwrap();
        assert_eq!(admission.inflight(), 1);
        drop(g2);
    }

    #[test]
    fn admission_model_lease_prevents_cross_model_batch() {
        let admission = Arc::new(Admission::new_with_capacity(8, Duration::from_secs(1), 2));
        let g1 = admission.acquire_for(true, Some("qwen3.5:7b")).unwrap();
        // Different model cannot share batch lanes, must wait for exclusive.
        let admission2 = Arc::clone(&admission);
        let handle =
            thread::spawn(move || admission2.acquire_for(true, Some("qwen3.5:14b")).unwrap());
        thread::sleep(Duration::from_millis(50));
        assert_eq!(admission.inflight(), 2); // 1 held + 1 queued due to model mismatch
        drop(g1);
        let g2 = handle.join().unwrap();
        assert_eq!(admission.inflight(), 1);
        drop(g2);
    }

    #[test]
    fn batch_eligibility_conservative_checks() {
        // Eligible: tp=1 qwen with one plain user string.
        let body =
            serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":"hi"}]});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Eligible: absent messages (prompt path) for Qwen.
        let body = serde_json::json!({"model":"qwen3.5:7b","prompt":"hi"});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Eligible: empty messages array for Qwen.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[]});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Tools disqualify.
        let body = serde_json::json!({"model":"qwen3.5:7b","tools":[{"type":"function","function":{"name":"x"}}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Image / multipart content disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:"}}]}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Multipart text-only array content also disqualifies (must be plain string).
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // system+user history disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[
            {"role":"system","content":"be brief"},
            {"role":"user","content":"hi"}
        ]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // user+assistant multi-turn disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":"hello"}
        ]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Tool-call content on the sole message disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[{
            "role":"user",
            "content":"hi",
            "tool_calls":[{"id":"c0","type":"function","function":{"name":"x","arguments":"{}"}}]
        }]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Qwen tp=4 pure EP is eligible when daemon admits batch.
        let body =
            serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":"hi"}]});
        assert!(is_batch_eligible_request(
            &body,
            Some(4),
            Some("qwen35"),
            true
        ));
        // Qwen tp=2 (and any non-1/non-4) disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b"});
        assert!(!is_batch_eligible_request(
            &body,
            Some(2),
            Some("qwen35"),
            true
        ));
        // Non-qwen disqualifies.
        let body = serde_json::json!({"model":"deepseek4:671b"});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("deepseek4"),
            true
        ));
        // Daemon load says batch incapable: HTTP admission must not invent it.
        let body =
            serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":"hi"}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            false
        ));
        // Eligible: tp=1 LFM2 dense with one plain user string when daemon admits batch.
        let body =
            serde_json::json!({"model":"lfm2.5:1.2b","messages":[{"role":"user","content":"hi"}]});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // Eligible: absent messages for LFM.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","prompt":"hi"});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // Eligible: empty messages for LFM.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[]});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM rejects system+user the same way.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[
            {"role":"system","content":"be brief"},
            {"role":"user","content":"hi"}
        ]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM rejects user+assistant.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":"hello"}
        ]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM rejects tool-call content.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[{
            "role":"user",
            "content":"hi",
            "tool_calls":[{"id":"c0","type":"function","function":{"name":"x","arguments":"{}"}}]
        }]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM rejects image/multipart content.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[{"role":"user","content":[
            {"type":"text","text":"describe"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,YWJj"}}
        ]}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM tp=4 disqualifies (dense remains tp=1 only).
        let body =
            serde_json::json!({"model":"lfm2.5:1.2b","messages":[{"role":"user","content":"hi"}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(4),
            Some("lfm2"),
            true
        ));
        // Daemon load says batch incapable: LFM HTTP admission must not invent it.
        let body =
            serde_json::json!({"model":"lfm2.5:1.2b","messages":[{"role":"user","content":"hi"}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            false
        ));
    }

    #[test]
    fn batch_messages_shape_matches_daemon_contract() {
        // Absent / empty → eligible shape.
        assert!(batch_messages_are_single_user(&serde_json::json!({})));
        assert!(batch_messages_are_single_user(
            &serde_json::json!({"messages":[]})
        ));
        // Exactly one plain user string → eligible.
        assert!(batch_messages_are_single_user(&serde_json::json!({
            "messages":[{"role":"user","content":"hi"}]
        })));
        // system+user, user+assistant → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[
                {"role":"system","content":"sys"},
                {"role":"user","content":"hi"}
            ]
        })));
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":"yo"}
            ]
        })));
        // Non-user sole role → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[{"role":"system","content":"sys"}]
        })));
        // tool_calls payload → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[{
                "role":"user",
                "content":"hi",
                "tool_calls":[{"id":"c0"}]
            }]
        })));
        // Multipart / image array content → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]
        })));
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"data:"}}
            ]}]
        })));
        // Non-array messages → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":"not-an-array"
        })));
    }

    #[test]
    fn qwen_mq4r_decode_prewarm_is_fail_closed_to_the_exact_route() {
        let qwen = serde_json::json!({ "arch": "qwen3_5_moe" });
        let qwen_dense = serde_json::json!({ "arch": "qwen3_5" });
        let deepseek = serde_json::json!({ "arch": "deepseek4" });

        assert!(should_prewarm_qwen_mq4r_decode(
            Path::new("qwen3.6-35b-a3b.mq4r"),
            &qwen,
            None,
        ));
        assert!(should_prewarm_qwen_mq4r_decode(
            Path::new("QWEN3.6-9B.MQ4R"),
            &qwen_dense,
            Some(1),
        ));
        assert!(!should_prewarm_qwen_mq4r_decode(
            Path::new("qwen3.6-35b-a3b.mq4"),
            &qwen,
            None,
        ));
        assert!(!should_prewarm_qwen_mq4r_decode(
            Path::new("deepseek-v4-flash.mq4r"),
            &deepseek,
            None,
        ));
        assert!(!should_prewarm_qwen_mq4r_decode(
            Path::new("qwen3.6-35b-a3b.mq4r"),
            &qwen,
            Some(2),
        ));
    }
}
