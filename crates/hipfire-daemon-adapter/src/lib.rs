// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Async daemon JSONL process adapter.

/// Re-exported so resource-lock status consumers (admin API, TUI) can match the
/// live flock state without a direct `hipfire-lock` dependency.
pub use hipfire_lock::LockState;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::future::BoxFuture;
use hipfire_daemon_protocol::{
    CollectRequest, CollectResponse, DaemonRequest, DaemonResponse, KldChunkEvent, KldEvalRequest,
    KldEvalResponse, RequestControl,
};
use hipfire_generate::{DoneEvent, GenerateTextRequest, ToolCall};
use hipfire_model::{
    AcceleratorInventory, LlmModelRegistry, ModelLoadParams, ModelLoadRequest, ModelLoadedResponse,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tracing::debug;

trait DaemonTransport: Send {
    #[cfg(test)]
    fn as_any(&self) -> &dyn std::any::Any;
    fn send_json<'a>(&'a mut self, req: &'a DaemonRequest) -> BoxFuture<'a, anyhow::Result<()>>;
    fn recv_response<'a>(&'a mut self) -> BoxFuture<'a, anyhow::Result<DaemonResponse>>;
}

struct StdioTransport {
    _child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl StdioTransport {
    async fn spawn(bin: &Path) -> anyhow::Result<Self> {
        let mut child = Command::new(bin)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| anyhow::anyhow!("failed to spawn daemon at {}: {e}", bin.display()))?;

        let stdin = BufWriter::new(child.stdin.take().expect("piped stdin"));
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));

        Ok(Self {
            _child: child,
            stdin,
            stdout,
        })
    }
}

impl DaemonTransport for StdioTransport {
    #[cfg(test)]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn send_json<'a>(&'a mut self, req: &'a DaemonRequest) -> BoxFuture<'a, anyhow::Result<()>> {
        Box::pin(async move {
            let line = serde_json::to_string(req)?;
            debug!("> {line}");
            self.stdin.write_all(line.as_bytes()).await?;
            self.stdin.write_all(b"\n").await?;
            self.stdin.flush().await?;
            Ok(())
        })
    }

    fn recv_response<'a>(&'a mut self) -> BoxFuture<'a, anyhow::Result<DaemonResponse>> {
        Box::pin(async move {
            let mut line = String::new();
            self.stdout.read_line(&mut line).await?;
            if line.is_empty() {
                anyhow::bail!("daemon stdout closed unexpectedly");
            }
            let line = line.trim_end();
            debug!("< {line}");
            Ok(serde_json::from_str(line)?)
        })
    }
}

pub struct DaemonEngine {
    transport: Box<dyn DaemonTransport>,
    pub worker_key_id: Option<String>,
}

pub struct GenerateCollected {
    pub text: String,
    pub done: DoneEvent,
    pub tool_calls: Vec<ToolCall>,
}

pub enum GenerateStreamEvent {
    Token(String),
    ToolCalls(Vec<ToolCall>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerateStreamControl {
    Continue,
    Cancel,
}

impl DaemonEngine {
    pub async fn spawn(bin: &Path) -> anyhow::Result<Self> {
        let transport = StdioTransport::spawn(bin).await?;
        Ok(Self {
            transport: Box::new(transport),
            worker_key_id: None,
        })
    }

    async fn send(&mut self, req: &DaemonRequest) -> anyhow::Result<()> {
        self.transport.send_json(req).await
    }

    async fn recv(&mut self) -> anyhow::Result<DaemonResponse> {
        self.transport.recv_response().await
    }

    /// Ask the daemon to abort a running request.
    ///
    /// This is fire-and-forget by protocol design: the matching generate
    /// stream is expected to drain its own terminal `done`/`error` event.
    pub async fn abort(&mut self, request_id: impl Into<String>) -> anyhow::Result<()> {
        self.send(&DaemonRequest::Abort(RequestControl {
            id: request_id.into(),
        }))
        .await
    }

    /// Ask the daemon to close an active thinking block and answer.
    ///
    /// Like `abort`, this does not wait for a separate acknowledgement; the
    /// active generate stream remains the authoritative response path.
    pub async fn force_answer(&mut self, request_id: impl Into<String>) -> anyhow::Result<()> {
        self.send(&DaemonRequest::ForceAnswer(RequestControl {
            id: request_id.into(),
        }))
        .await
    }

    /// Send `load` and wait for `loaded`.
    pub async fn load(
        &mut self,
        model_path: &str,
        params: ModelLoadParams,
    ) -> anyhow::Result<ModelLoadedResponse> {
        let request_id = uuid::Uuid::new_v4().to_string();
        self.send(&DaemonRequest::Load(ModelLoadRequest {
            model: model_path.to_string(),
            params,
            request_id: Some(request_id.clone()),
        }))
        .await?;
        let expected_response = Some(request_id);

        loop {
            match self.recv().await? {
                DaemonResponse::Loaded(r) => {
                    if let Some(expected) = &expected_response {
                        if matches!(r.response_id.as_deref(), Some(actual) if actual != expected) {
                            tracing::warn!(
                                "stale load response: got response_id={:?} expected={:?}",
                                r.response_id,
                                expected_response
                            );
                            continue;
                        }
                    }
                    self.worker_key_id = Some(r.worker_key_id.clone());
                    return Ok(r);
                }
                DaemonResponse::Error(e) => anyhow::bail!("daemon load error: {}", e.message),
                DaemonResponse::Unknown => {}
                other => {
                    tracing::warn!("unexpected response during load: {other:?}");
                }
            }
        }
    }

    /// Send `unload` and wait for `unloaded`.
    pub async fn unload(&mut self) -> anyhow::Result<()> {
        self.send(&DaemonRequest::Unload).await?;
        loop {
            match self.recv().await? {
                DaemonResponse::Unloaded => {
                    self.worker_key_id = None;
                    return Ok(());
                }
                DaemonResponse::Error(e) => anyhow::bail!("daemon unload error: {}", e.message),
                DaemonResponse::Unknown => {}
                other => {
                    tracing::warn!("unexpected response during unload: {other:?}");
                }
            }
        }
    }

    /// Send `reset` and wait for the daemon to confirm state reset.
    pub async fn reset(&mut self) -> anyhow::Result<()> {
        self.send(&DaemonRequest::Reset).await?;
        loop {
            match self.recv().await? {
                DaemonResponse::Reset => return Ok(()),
                DaemonResponse::Error(e) => anyhow::bail!("daemon reset error: {}", e.message),
                DaemonResponse::Unknown => {}
                other => {
                    tracing::warn!("unexpected response during reset: {other:?}");
                }
            }
        }
    }

    /// Send `ping` and wait for `pong`.
    pub async fn ping(&mut self) -> anyhow::Result<()> {
        self.send(&DaemonRequest::Ping).await?;
        loop {
            match self.recv().await? {
                DaemonResponse::Pong => return Ok(()),
                DaemonResponse::Unknown => {}
                other => {
                    tracing::warn!("unexpected response during ping: {other:?}");
                }
            }
        }
    }

    /// Send `inventory` and wait for accelerator inventory.
    pub async fn inventory(&mut self) -> anyhow::Result<AcceleratorInventory> {
        self.send(&DaemonRequest::Inventory).await?;
        loop {
            match self.recv().await? {
                DaemonResponse::Inventory(inventory) => return Ok(inventory),
                DaemonResponse::Error(e) => anyhow::bail!("daemon inventory error: {}", e.message),
                DaemonResponse::Unknown => {}
                other => {
                    tracing::warn!("unexpected response during inventory: {other:?}");
                }
            }
        }
    }

    /// Send `model_registry` and wait for the daemon's startup model inventory.
    pub async fn model_registry(&mut self) -> anyhow::Result<LlmModelRegistry> {
        self.send(&DaemonRequest::ModelRegistry).await?;
        loop {
            match self.recv().await? {
                DaemonResponse::ModelRegistry { registry } => return Ok(registry),
                DaemonResponse::Error(e) => {
                    anyhow::bail!("daemon model_registry error: {}", e.message)
                }
                DaemonResponse::Unknown => {}
                other => {
                    tracing::warn!("unexpected response during model_registry: {other:?}");
                }
            }
        }
    }

    /// Send `collect` (calibrate the resident model in place) and wait for the
    /// resulting `.calib.hfq` path + summary.
    pub async fn collect(&mut self, req: CollectRequest) -> anyhow::Result<CollectResponse> {
        self.send(&DaemonRequest::Collect(req)).await?;
        loop {
            match self.recv().await? {
                DaemonResponse::Collected(resp) => return Ok(resp),
                DaemonResponse::Error(e) => anyhow::bail!("daemon collect error: {}", e.message),
                DaemonResponse::Unknown => {}
                other => {
                    tracing::warn!("unexpected response during collect: {other:?}");
                }
            }
        }
    }

    /// Send `kld_eval` (build a KLD reference and/or score the resident model
    /// against one, with no reload) and wait for the final result. Per-chunk
    /// `KldChunk` progress frames are passed to `on_chunk` as they stream.
    pub async fn kld_eval(
        &mut self,
        req: KldEvalRequest,
        mut on_chunk: impl FnMut(&KldChunkEvent),
    ) -> anyhow::Result<KldEvalResponse> {
        self.send(&DaemonRequest::KldEval(req)).await?;
        loop {
            match self.recv().await? {
                DaemonResponse::KldChunk(ev) => on_chunk(&ev),
                DaemonResponse::KldEvaled(resp) => return Ok(resp),
                DaemonResponse::Error(e) => anyhow::bail!("daemon kld_eval error: {}", e.message),
                DaemonResponse::Unknown => {}
                other => tracing::warn!("unexpected response during kld_eval: {other:?}"),
            }
        }
    }

    /// Send `generate` and collect all tokens. Returns (text, done).
    pub async fn generate(
        &mut self,
        req: GenerateTextRequest,
    ) -> anyhow::Result<(String, DoneEvent)> {
        let collected = self.generate_collected(req).await?;
        Ok((collected.text, collected.done))
    }

    /// Send `generate` and collect all text plus structured tool-call events.
    pub async fn generate_collected(
        &mut self,
        req: GenerateTextRequest,
    ) -> anyhow::Result<GenerateCollected> {
        let request_id = req.id.clone();
        self.send(&DaemonRequest::Generate(req)).await?;
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        loop {
            match self.recv().await? {
                DaemonResponse::Token(t) => {
                    if t.id == request_id {
                        text.push_str(&t.text)
                    }
                }
                DaemonResponse::ToolCalls(t) => {
                    if t.id == request_id {
                        tool_calls.extend(t.calls);
                    }
                }
                DaemonResponse::Done(d) => {
                    if d.id == request_id {
                        return Ok(GenerateCollected {
                            text,
                            done: d,
                            tool_calls,
                        });
                    }
                    tracing::warn!(
                        "stale done response: got id={} expected={}",
                        d.id,
                        request_id
                    );
                }
                DaemonResponse::Error(e) => anyhow::bail!("daemon generate error: {}", e.message),
                DaemonResponse::Unknown => {}
                other => {
                    tracing::warn!("unexpected response during generate: {other:?}");
                }
            }
        }
    }

    /// Send `generate` and stream tokens via a callback. Returns done.
    pub async fn generate_streaming<F>(
        &mut self,
        req: GenerateTextRequest,
        mut on_token: F,
    ) -> anyhow::Result<DoneEvent>
    where
        F: FnMut(String),
    {
        self.generate_streaming_events(req, move |event| {
            if let GenerateStreamEvent::Token(text) = event {
                on_token(text);
            }
        })
        .await
    }

    /// Send `generate` and stream typed generation events via a callback. Returns done.
    pub async fn generate_streaming_events<F>(
        &mut self,
        req: GenerateTextRequest,
        mut on_event: F,
    ) -> anyhow::Result<DoneEvent>
    where
        F: FnMut(GenerateStreamEvent),
    {
        self.generate_streaming_events_controlled(req, move |event| {
            on_event(event);
            GenerateStreamControl::Continue
        })
        .await?
        .ok_or_else(|| anyhow::anyhow!("generation cancelled"))
    }

    /// Send `generate` and stream typed events until completion or caller cancellation.
    ///
    /// Returning `GenerateStreamControl::Cancel` stops reading the daemon
    /// stream and returns `Ok(None)`. Callers must then discard this engine:
    /// unread daemon events would otherwise corrupt the next request.
    pub async fn generate_streaming_events_controlled<F>(
        &mut self,
        req: GenerateTextRequest,
        mut on_event: F,
    ) -> anyhow::Result<Option<DoneEvent>>
    where
        F: FnMut(GenerateStreamEvent) -> GenerateStreamControl,
    {
        let request_id = req.id.clone();
        self.send(&DaemonRequest::Generate(req)).await?;
        loop {
            match self.recv().await? {
                DaemonResponse::Token(t) => {
                    if t.id == request_id {
                        if on_event(GenerateStreamEvent::Token(t.text))
                            == GenerateStreamControl::Cancel
                        {
                            return Ok(None);
                        }
                    }
                }
                DaemonResponse::ToolCalls(t) => {
                    if t.id == request_id {
                        if on_event(GenerateStreamEvent::ToolCalls(t.calls))
                            == GenerateStreamControl::Cancel
                        {
                            return Ok(None);
                        }
                    }
                }
                DaemonResponse::Done(d) => {
                    if d.id == request_id {
                        return Ok(Some(d));
                    }
                    tracing::warn!(
                        "stale done response: got id={} expected={}",
                        d.id,
                        request_id
                    );
                }
                DaemonResponse::Error(e) => anyhow::bail!("daemon generate error: {}", e.message),
                DaemonResponse::Unknown => {}
                other => {
                    tracing::warn!("unexpected response during generate: {other:?}");
                }
            }
        }
    }
}

/// Locate the daemon binary. Priority:
/// 1. `HIPFIRE_DAEMON_BIN` env var
/// 2. `~/.hipfire/bin/daemon`
/// 3. repo-root `target/release/hipfire-daemon`
/// 4. repo-root `target/debug/hipfire-daemon`
pub fn find_daemon_bin() -> Option<PathBuf> {
    find_daemon_bin_candidates()
        .into_iter()
        .find(|p| p.exists())
}

pub fn find_daemon_bin_or_error() -> anyhow::Result<PathBuf> {
    find_daemon_bin().ok_or_else(|| {
        anyhow::anyhow!(
            "daemon binary not found; build with: cargo build -p hipfire-daemon --bin hipfire-daemon"
        )
    })
}

fn find_daemon_bin_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(p) = std::env::var("HIPFIRE_DAEMON_BIN") {
        candidates.push(PathBuf::from(p));
    }

    if let Some(home) = dirs::home_dir() {
        let hipfire_bin = home.join(".hipfire").join("bin");
        for name in &["hipfire-daemon", "daemon"] {
            candidates.push(hipfire_bin.join(name));
        }
    }

    let exe = std::env::consts::EXE_SUFFIX;
    let repo = repo_root().unwrap_or_else(|| PathBuf::from("."));
    for rel in &[
        format!("target/release/hipfire-daemon{exe}"),
        format!("target/debug/hipfire-daemon{exe}"),
    ] {
        candidates.push(repo.join(rel));
    }

    candidates
}

fn repo_root() -> Option<PathBuf> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok();
    if let Some(out) = out {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return Some(PathBuf::from(s));
            }
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fallback = manifest_dir.join("../..");
    if fallback.join("Cargo.toml").exists() {
        fallback.canonicalize().ok().or(Some(fallback))
    } else {
        None
    }
}

/// Held GPU/NPU/CPU resource leases for the daemon's lifetime. Each entry is a
/// `flock(2)` guard ([`hipfire_lock::FlockGuard`]); dropping the lease closes the
/// fds, and the kernel releases the locks — including on SIGKILL / crash, so there
/// are no stale leases to reclaim (unlike the old mkdir+owner.json scheme). The
/// single-GPU lease shares [`hipfire_lock::gpu_lock_path`]'s inode with the
/// `hipfire gpu-lock` CLI, so daemon and non-daemon GPU users mutually exclude.
#[derive(Debug)]
pub struct ResourceLease {
    #[allow(dead_code)] // held purely for its Drop (releases the flocks)
    guards: Vec<hipfire_lock::FlockGuard>,
}

pub fn sanitize_resource_id(id: &str) -> String {
    let mut out = String::with_capacity(id.len().max(1));
    for ch in id.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

fn parse_csv_ids(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

pub fn parse_cpu_core_list(raw: Option<String>) -> Result<Vec<usize>, String> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    let mut out = BTreeSet::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((start, end)) = trimmed.split_once('-') {
            let start = start
                .parse::<usize>()
                .map_err(|_| format!("invalid CPU core id: {trimmed}"))?;
            let end = end
                .parse::<usize>()
                .map_err(|_| format!("invalid CPU core id: {trimmed}"))?;
            if end < start {
                return Err(format!("invalid CPU core range: {trimmed}"));
            }
            for core in start..=end {
                out.insert(core);
            }
        } else {
            out.insert(
                trimmed
                    .parse::<usize>()
                    .map_err(|_| format!("invalid CPU core id: {trimmed}"))?,
            );
        }
    }
    Ok(out.into_iter().collect())
}

fn resolve_visible_hip_ids() -> Vec<String> {
    std::env::var("HIP_VISIBLE_DEVICES")
        .ok()
        .or_else(|| std::env::var("ROCR_VISIBLE_DEVICES").ok())
        .map(|raw| parse_csv_ids(&raw))
        .filter(|ids| !ids.is_empty())
        .unwrap_or_default()
}

pub fn resolve_hip_lock_ids() -> Vec<String> {
    let visible = resolve_visible_hip_ids();
    if let Ok(raw) = std::env::var("HIPFIRE_DEVICES") {
        let ids = parse_csv_ids(&raw);
        if !ids.is_empty() {
            return ids
                .into_iter()
                .map(|id| {
                    id.parse::<usize>()
                        .ok()
                        .and_then(|idx| visible.get(idx).cloned())
                        .unwrap_or(id)
                })
                .collect();
        }
    }
    visible.into_iter().next().into_iter().collect::<Vec<_>>()
}

fn discover_npu_lock_ids() -> Vec<String> {
    let mut ids = BTreeSet::new();
    for root in ["/sys/class/accel", "/dev/accel"] {
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    ids.insert(name.to_string());
                }
            }
        }
    }
    ids.into_iter().collect()
}

fn resolve_npu_lock_ids() -> Vec<String> {
    // HIPFIRE_RESOURCE_LOCK_NPUS=1 leases every detected NPU; comma lists lease explicit NPU IDs.
    let Ok(raw) = std::env::var("HIPFIRE_RESOURCE_LOCK_NPUS") else {
        return Vec::new();
    };
    let trimmed = raw.trim();
    if matches!(
        trimmed,
        "" | "0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO"
    ) {
        return Vec::new();
    }
    if matches!(trimmed, "1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES") {
        return discover_npu_lock_ids();
    }
    parse_csv_ids(trimmed)
}

pub fn resource_lock_requests() -> Result<Vec<String>, String> {
    let mut resources = Vec::new();
    let hip_ids = resolve_hip_lock_ids();
    if hip_ids.is_empty() {
        resources.push("hip-gpu-0".to_string());
    } else {
        resources.extend(
            hip_ids
                .into_iter()
                .map(|id| format!("hip-gpu-{}", sanitize_resource_id(&id))),
        );
    }
    resources.extend(
        resolve_npu_lock_ids()
            .into_iter()
            .map(|id| format!("npu-{}", sanitize_resource_id(&id))),
    );
    resources.extend(
        // HIPFIRE_RESOURCE_LOCK_CPU_CORES=0,2-4 adds daemon startup leases for CPU cores.
        parse_cpu_core_list(std::env::var("HIPFIRE_RESOURCE_LOCK_CPU_CORES").ok())?
            .into_iter()
            .map(|core| format!("cpu-core-{core}")),
    );
    resources.sort();
    resources.dedup();
    Ok(resources)
}

fn current_hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .or_else(|_| std::fs::read_to_string("/etc/hostname"))
        .map(|s| s.trim().to_string())
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Lockfile path for a daemon resource lease. The single GPU resource (the common
/// case) shares [`hipfire_lock::gpu_lock_path`] with the `hipfire gpu-lock` CLI —
/// same inode, so daemon and non-daemon GPU users mutually exclude via ONE flock.
/// Multi-GPU / NPU / CPU resources get their own flock file under `root`.
pub fn resource_lock_path(root: &Path, resource: &str, gpu_resource_count: usize) -> PathBuf {
    if resource.starts_with("hip-gpu-") && gpu_resource_count <= 1 {
        hipfire_lock::gpu_lock_path()
    } else {
        root.join(format!("{resource}.lock"))
    }
}

/// Holder line written into a lease lockfile (under the held flock), mirroring the
/// `hipfire gpu-lock` CLI format so `status`/`probe` show a consistent owner.
fn lease_holder_line(resource: &str) -> String {
    format!(
        "daemon resource={resource} pid={} host={} acquired_epoch={}",
        std::process::id(),
        current_hostname(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    )
}

/// Probe every resource lease lockfile and report (name, path, live flock state).
/// Enumerates the shared GPU lock plus any per-resource flock files under `root`,
/// so the report reflects what is *actually* held right now (kernel `flock` probe),
/// not a stale lockfile — used by the admin API and the TUI status view.
pub fn resource_lock_report(root: &Path) -> Vec<(String, PathBuf, hipfire_lock::LockState)> {
    let mut out: Vec<(String, PathBuf, hipfire_lock::LockState)> = Vec::new();
    let gpu = hipfire_lock::gpu_lock_path();
    if gpu.exists() {
        let st = hipfire_lock::probe(&gpu).unwrap_or(hipfire_lock::LockState::Free);
        out.push(("gpu".to_string(), gpu.clone(), st));
    }
    if let Ok(entries) = std::fs::read_dir(root) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("lock") {
                let name = p
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                let st = hipfire_lock::probe(&p).unwrap_or(hipfire_lock::LockState::Free);
                out.push((name, p, st));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

pub fn acquire_resource_lease_or_exit() -> ResourceLease {
    // HIPFIRE_RESOURCE_LOCK=0 disables daemon startup resource leases.
    if std::env::var("HIPFIRE_RESOURCE_LOCK").ok().as_deref() == Some("0") {
        return ResourceLease { guards: Vec::new() };
    }

    let resources = match resource_lock_requests() {
        Ok(resources) => resources,
        Err(e) => {
            eprintln!("FATAL: invalid hipfire resource lock config: {e}");
            std::process::exit(1);
        }
    };
    if resources.is_empty() {
        return ResourceLease { guards: Vec::new() };
    }

    // HIPFIRE_RESOURCE_LOCK_DIR overrides the per-resource flock-file root.
    let root = std::env::var("HIPFIRE_RESOURCE_LOCK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("hipfire-resource-locks"));
    // HIPFIRE_RESOURCE_LOCK_WAIT_MS waits for busy leases before failing startup.
    // 0 = fail-fast (single try); >0 = block up to that many ms per resource.
    let wait_ms = std::env::var("HIPFIRE_RESOURCE_LOCK_WAIT_MS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(0);
    let timeout = (wait_ms > 0).then(|| Duration::from_millis(wait_ms));
    let gpu_count = resources.iter().filter(|r| r.starts_with("hip-gpu-")).count();
    let mut guards: Vec<hipfire_lock::FlockGuard> = Vec::new();

    for resource in &resources {
        let path = resource_lock_path(&root, resource, gpu_count);
        let mut guard = match hipfire_lock::FlockGuard::open(&path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("FATAL: open resource lockfile {}: {e}", path.display());
                std::process::exit(1); // `guards` drops → kernel releases held flocks
            }
        };
        // wait_ms==0 → single try_lock (fail-fast); wait_ms>0 → block up to timeout.
        let acquired = match timeout {
            Some(t) => guard.lock_blocking(Duration::from_millis(250), Some(t), |holder| {
                eprintln!(
                    "[hipfire] waiting for resource {resource} (held by {})",
                    if holder.is_empty() { "another process" } else { holder }
                );
            }),
            None => guard.try_lock(),
        };
        match acquired {
            Ok(true) => {
                let _ = guard.write_holder(&lease_holder_line(resource));
                guards.push(guard);
            }
            Ok(false) => {
                let holder = match hipfire_lock::probe(&path) {
                    Ok(hipfire_lock::LockState::Busy(h)) => h,
                    _ => String::new(),
                };
                eprintln!(
                    "FATAL: hipfire resource {resource} ({}) is locked by {}",
                    path.display(),
                    if holder.is_empty() { "another process" } else { &holder }
                );
                eprintln!(
                    "Set HIPFIRE_RESOURCE_LOCK_WAIT_MS to wait, or HIPFIRE_RESOURCE_LOCK=0 to bypass."
                );
                std::process::exit(1); // `guards` drops → kernel releases held flocks
            }
            Err(e) => {
                eprintln!("FATAL: lock resource {resource} ({}): {e}", path.display());
                std::process::exit(1);
            }
        }
    }
    eprintln!(
        "[hipfire] resource locks acquired (flock): {}",
        resources.join(", ")
    );
    ResourceLease { guards }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_generate::GenerationSamplingPolicy;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn temp_lock_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "hipfire-daemon-lock-test-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    struct MockTransport {
        sent: Vec<String>,
        responses: VecDeque<DaemonResponse>,
    }

    impl DaemonTransport for MockTransport {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn send_json<'a>(
            &'a mut self,
            req: &'a DaemonRequest,
        ) -> BoxFuture<'a, anyhow::Result<()>> {
            Box::pin(async move {
                self.sent.push(serde_json::to_string(req)?);
                Ok(())
            })
        }

        fn recv_response<'a>(&'a mut self) -> BoxFuture<'a, anyhow::Result<DaemonResponse>> {
            Box::pin(async move {
                self.responses
                    .pop_front()
                    .ok_or_else(|| anyhow::anyhow!("mock response queue exhausted"))
            })
        }
    }

    fn mock_engine(responses: Vec<DaemonResponse>) -> DaemonEngine {
        DaemonEngine {
            transport: Box::new(MockTransport {
                sent: Vec::new(),
                responses: responses.into(),
            }),
            worker_key_id: None,
        }
    }

    #[test]
    fn daemon_binary_candidates_include_env_home_and_repo_targets() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("HIPFIRE_DAEMON_BIN", "/tmp/custom-hipfire-daemon");
        }
        let candidates = find_daemon_bin_candidates();
        unsafe {
            std::env::remove_var("HIPFIRE_DAEMON_BIN");
        }

        assert_eq!(candidates[0], PathBuf::from("/tmp/custom-hipfire-daemon"));
        assert!(candidates
            .iter()
            .any(|path| path.ends_with(".hipfire/bin/hipfire-daemon")));
        assert!(candidates
            .iter()
            .any(|path| path.ends_with("target/release/hipfire-daemon")));
        assert!(candidates
            .iter()
            .any(|path| path.ends_with("target/debug/hipfire-daemon")));
    }

    #[tokio::test]
    async fn load_ignores_stale_response_id_and_records_worker() {
        let mut engine = mock_engine(vec![
            DaemonResponse::Loaded(ModelLoadedResponse {
                worker_key_id: "stale-worker".to_string(),
                arch: None,
                cache_capable: None,
                dim: None,
                layers: None,
                vocab: None,
                model_worker: None,
                response_id: Some("stale".to_string()),
            }),
            DaemonResponse::Loaded(ModelLoadedResponse {
                worker_key_id: "worker-a".to_string(),
                arch: Some("qwen35".to_string()),
                cache_capable: Some(true),
                dim: Some(4096),
                layers: Some(32),
                vocab: Some(151936),
                model_worker: None,
                response_id: None,
            }),
        ]);

        let loaded = engine
            .load("model.hfq", ModelLoadParams::default())
            .await
            .unwrap();
        assert_eq!(loaded.worker_key_id, "worker-a");
        assert_eq!(engine.worker_key_id.as_deref(), Some("worker-a"));
    }

    #[tokio::test]
    async fn inventory_returns_shared_accelerator_contract() {
        let mut engine = mock_engine(vec![DaemonResponse::Inventory(
            AcceleratorInventory::from_devices(
                "daemon",
                vec![hipfire_model::AcceleratorDeviceInfo::hip(
                    "0",
                    0,
                    Some("gfx1201".to_string()),
                    Some(24_000_000_000),
                    Some(false),
                    Some("HIP 6.4".to_string()),
                )],
            ),
        )]);

        let inventory = engine.inventory().await.unwrap();
        assert_eq!(inventory.source, "daemon");
        assert_eq!(inventory.devices.len(), 1);
        assert_eq!(inventory.devices[0].device_id, "0");
        assert_eq!(inventory.devices[0].device_class(), "discrete");
    }

    #[tokio::test]
    async fn generate_collects_only_matching_tokens_until_matching_done() {
        let mut engine = mock_engine(vec![
            DaemonResponse::Token(hipfire_generate::TokenEvent {
                id: "other".to_string(),
                text: "skip".to_string(),
            }),
            DaemonResponse::Token(hipfire_generate::TokenEvent {
                id: "req-1".to_string(),
                text: "hello".to_string(),
            }),
            DaemonResponse::Token(hipfire_generate::TokenEvent {
                id: "req-1".to_string(),
                text: " world".to_string(),
            }),
            DaemonResponse::ToolCalls(hipfire_generate::ToolCallsEvent {
                id: "other".to_string(),
                calls: vec![hipfire_generate::ToolCall {
                    name: "skip".to_string(),
                    arguments: serde_json::json!({}),
                }],
            }),
            DaemonResponse::ToolCalls(hipfire_generate::ToolCallsEvent {
                id: "req-1".to_string(),
                calls: vec![hipfire_generate::ToolCall {
                    name: "lookup".to_string(),
                    arguments: serde_json::json!({"q": "hipfire"}),
                }],
            }),
            DaemonResponse::Done(DoneEvent {
                id: "req-1".to_string(),
                tokens: 2,
                tok_s: None,
                prefill_tokens: None,
                prefill_ms: None,
                prefill_tok_s: None,
                decode_tok_s: None,
                ttft_ms: None,
                finish_reason: Some("stop".to_string()),
                response_id: None,
                extra: Default::default(),
            }),
        ]);

        let req = GenerateTextRequest {
            id: "req-1".to_string(),
            prompt: "hello".to_string(),
            messages: None,
            sampling: GenerationSamplingPolicy {
                temperature: 0.7,
                max_tokens: 8,
                top_p: None,
                repeat_penalty: None,
            },
            worker_key_id: Some("worker-a".to_string()),
            tools: None,
            system: None,
            stop: None,
            image_base64: None,
            thinking: None,
            thinking_mode: None,
            reasoning_effort: None,
            assistant_prefix: None,
            max_think_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            request_id: None,
            evidence_dir: None,
        };
        let collected = engine.generate_collected(req).await.unwrap();
        assert_eq!(collected.text, "hello world");
        assert_eq!(collected.done.tokens, 2);
        assert_eq!(collected.tool_calls.len(), 1);
        assert_eq!(collected.tool_calls[0].name, "lookup");
        assert_eq!(
            collected.tool_calls[0].arguments,
            serde_json::json!({"q": "hipfire"})
        );
    }

    #[tokio::test]
    async fn reset_waits_for_reset_response() {
        let mut engine = mock_engine(vec![DaemonResponse::Reset]);
        engine.reset().await.unwrap();
    }

    #[tokio::test]
    async fn request_control_helpers_send_bun_wire_shape_without_waiting() {
        let mut engine = mock_engine(vec![]);

        engine.abort("req-1").await.unwrap();
        engine.force_answer("req-1").await.unwrap();

        let transport = engine
            .transport
            .as_any()
            .downcast_ref::<MockTransport>()
            .expect("mock transport");
        assert_eq!(
            transport.sent,
            vec![
                r#"{"type":"abort","id":"req-1"}"#,
                r#"{"type":"force_answer","id":"req-1"}"#,
            ]
        );
    }

    #[tokio::test]
    async fn generate_streaming_events_forwards_tokens_and_tool_calls() {
        let mut engine = mock_engine(vec![
            DaemonResponse::Token(hipfire_generate::TokenEvent {
                id: "req-1".to_string(),
                text: "before".to_string(),
            }),
            DaemonResponse::ToolCalls(hipfire_generate::ToolCallsEvent {
                id: "req-1".to_string(),
                calls: vec![hipfire_generate::ToolCall {
                    name: "lookup".to_string(),
                    arguments: serde_json::json!({"q": "hipfire"}),
                }],
            }),
            DaemonResponse::Done(DoneEvent {
                id: "req-1".to_string(),
                tokens: 2,
                tok_s: None,
                prefill_tokens: None,
                prefill_ms: None,
                prefill_tok_s: None,
                decode_tok_s: None,
                ttft_ms: None,
                finish_reason: Some("tool_calls".to_string()),
                response_id: None,
                extra: Default::default(),
            }),
        ]);

        let req = GenerateTextRequest::from_prompt(
            "req-1".to_string(),
            "hello",
            GenerationSamplingPolicy::greedy(8),
        );
        let mut seen = Vec::new();
        let done = engine
            .generate_streaming_events(req, |event| match event {
                GenerateStreamEvent::Token(text) => seen.push(format!("token:{text}")),
                GenerateStreamEvent::ToolCalls(calls) => {
                    seen.push(format!("tool:{}", calls[0].name))
                }
            })
            .await
            .unwrap();

        assert_eq!(seen, vec!["token:before", "tool:lookup"]);
        assert_eq!(done.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[tokio::test]
    async fn controlled_stream_can_stop_without_waiting_for_done() {
        let mut engine = mock_engine(vec![
            DaemonResponse::Token(hipfire_generate::TokenEvent {
                id: "req-1".to_string(),
                text: "first".to_string(),
            }),
            DaemonResponse::Token(hipfire_generate::TokenEvent {
                id: "req-1".to_string(),
                text: "unread".to_string(),
            }),
        ]);

        let req = GenerateTextRequest::from_prompt(
            "req-1".to_string(),
            "hello",
            GenerationSamplingPolicy::greedy(8),
        );
        let mut seen = Vec::new();
        let done = engine
            .generate_streaming_events_controlled(req, |event| {
                if let GenerateStreamEvent::Token(text) = event {
                    seen.push(text);
                }
                GenerateStreamControl::Cancel
            })
            .await
            .unwrap();

        assert!(done.is_none());
        assert_eq!(seen, vec!["first"]);
    }

    #[test]
    fn resource_lock_cpu_core_list_parser_matches_cli_shape() {
        assert_eq!(
            parse_cpu_core_list(Some("0,2-4,3".to_string())).unwrap(),
            vec![0, 2, 3, 4]
        );
        assert!(parse_cpu_core_list(Some("4-2".to_string()))
            .unwrap_err()
            .contains("invalid CPU core range"));
        assert!(parse_cpu_core_list(Some("gpu0".to_string()))
            .unwrap_err()
            .contains("invalid CPU core id"));
    }

    #[test]
    fn resource_lock_maps_logical_hipfire_devices_through_visible_devices() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("HIP_VISIBLE_DEVICES", "3,5");
            std::env::remove_var("ROCR_VISIBLE_DEVICES");
            std::env::set_var("HIPFIRE_DEVICES", "1");
        }
        assert_eq!(resolve_hip_lock_ids(), vec!["5".to_string()]);
        unsafe {
            std::env::remove_var("HIP_VISIBLE_DEVICES");
            std::env::remove_var("HIPFIRE_DEVICES");
        }
    }

    #[test]
    fn resource_lock_path_maps_single_gpu_to_cli_path() {
        // Single GPU → shares the canonical `hipfire gpu-lock` inode.
        let root = temp_lock_root("path-map");
        assert_eq!(
            resource_lock_path(&root, "hip-gpu-0", 1),
            hipfire_lock::gpu_lock_path()
        );
        // Multi-GPU / NPU / CPU → per-resource flock file under root.
        assert_eq!(
            resource_lock_path(&root, "hip-gpu-1", 2),
            root.join("hip-gpu-1.lock")
        );
        assert_eq!(
            resource_lock_path(&root, "npu-0", 1),
            root.join("npu-0.lock")
        );
    }

    #[test]
    fn resource_lock_flock_excludes_then_releases_on_drop() {
        // The flock guard is held while alive and released (kernel-level) on drop —
        // a second probe sees Busy, then Free after the first guard drops. No stale
        // lockfile to reclaim (the inode lock, not the file's existence, is the lock).
        let root = temp_lock_root("flock");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("hip-gpu-7.lock");

        let mut a = hipfire_lock::FlockGuard::open(&path).unwrap();
        assert!(a.try_lock().unwrap());
        a.write_holder(&lease_holder_line("hip-gpu-7")).unwrap();
        assert!(matches!(
            hipfire_lock::probe(&path).unwrap(),
            hipfire_lock::LockState::Busy(_)
        ));

        drop(a); // kernel releases — equivalent to the holding process dying
        assert!(matches!(
            hipfire_lock::probe(&path).unwrap(),
            hipfire_lock::LockState::Free
        ));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
