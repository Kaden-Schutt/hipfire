// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Synchronous typed transport for the hipfire engine daemon JSONL protocol.
//!
//! The daemon remains the GPU/runtime boundary. CLI, serve, TUI helpers, and
//! Python tooling can share this Rust transport instead of duplicating process
//! management and line framing.

use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread,
    time::Duration,
};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedDaemonError {
    pub message: String,
    pub class: String,
    pub retryable: bool,
    pub rolled_back: bool,
    /// Internal generation attempt correlation (not the public completion id).
    pub attempt_id: u64,
    pub id: Option<String>,
}

impl std::fmt::Display for TypedDaemonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{class} retryable={retryable} rolled_back={rolled_back} attempt={attempt}] {message}",
            class = self.class,
            retryable = self.retryable,
            rolled_back = self.rolled_back,
            attempt = self.attempt_id,
            message = self.message
        )
    }
}

/// Stable daemon error class strings (wire `class` field). Classification must
/// use these fields — never message-prefix matching.
pub mod error_class {
    pub const TRANSIENT: &str = "transient";
    pub const MALFORMED: &str = "malformed";
    pub const VALIDATION: &str = "validation";
    pub const CONTEXT_LENGTH: &str = "context_length";
    pub const CANCEL: &str = "cancel";
    pub const TRANSPORT: &str = "transport";
    pub const UNSUPPORTED: &str = "unsupported";
    pub const INTERNAL: &str = "internal";
    pub const ADAPTIVE_POISON: &str = "adaptive_poison";
    pub const DETERMINISTIC_MISMATCH: &str = "deterministic_mismatch";
}

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("daemon binary not found: {0}")]
    MissingDaemon(PathBuf),
    #[error("failed to spawn daemon {path}: {source}")]
    Spawn {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("daemon stdin is unavailable")]
    MissingStdin,
    #[error("daemon stdout is unavailable")]
    MissingStdout,
    #[error("daemon I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("daemon emitted invalid JSON: {message}; line={line}")]
    InvalidJson { message: String, line: String },
    #[error("daemon closed its output (status {status})")]
    Closed { status: String },
    #[error("daemon protocol error: {0}")]
    Protocol(String),
    #[error("daemon error: {0}")]
    Daemon(TypedDaemonError),
    #[error("HTTP service error: {0}")]
    Http(String),
    /// Client explicitly cancelled the stream; never retried client-side.
    #[error("stream cancelled")]
    Cancelled,
    /// SSE ended before an authoritative finish choice and `[DONE]`.
    #[error("premature SSE EOF: {0}")]
    PrematureEof(String),
}

impl ClientError {
    /// Typed daemon failure when present (for retry classification).
    pub fn typed_daemon(&self) -> Option<&TypedDaemonError> {
        match self {
            Self::Daemon(err) => Some(err),
            _ => None,
        }
    }
}

/// Return an address suitable for connecting to a server that may be bound to
/// a wildcard interface.
pub fn probe_host(host: &str) -> &str {
    match host {
        "0.0.0.0" | "" => "127.0.0.1",
        "::" => "::1",
        other => other,
    }
}

pub fn service_url(host: &str, port: u16, path: &str) -> String {
    let host = probe_host(host);
    if host.contains(':') {
        format!("http://[{host}]:{port}{path}")
    } else {
        format!("http://{host}:{port}{path}")
    }
}

pub fn service_ready(host: &str, port: u16, timeout: Duration) -> bool {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false)
        .build()
        .into();
    agent
        .get(&service_url(host, port, "/health"))
        .call()
        .is_ok_and(|response| response.status().is_success())
}

pub fn complete_openai_chat(
    host: &str,
    port: u16,
    mut body: Value,
    timeout: Duration,
) -> Result<Value> {
    body["stream"] = Value::Bool(false);
    let agent = http_agent(timeout);
    let mut response = agent
        .post(&service_url(host, port, "/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .send(body.to_string())
        .map_err(|error| ClientError::Http(error.to_string()))?;
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|error| ClientError::Http(error.to_string()))?;
    if status >= 400 {
        return Err(ClientError::Http(http_error_message(status, &text)));
    }
    serde_json::from_str(&text).map_err(|error| {
        ClientError::Http(format!(
            "service emitted invalid JSON: {error}; body={}",
            text.chars().take(512).collect::<String>()
        ))
    })
}

/// Typed OpenAI chat-completion SSE events consumed by first-party clients.
///
/// Wire markers / model tool protocol are never re-parsed here. Structured
/// tool calls arrive only as authoritative OpenAI delta fields already lowered
/// by the server. Client-side retry is intentionally unsupported.
#[derive(Debug, Clone, PartialEq)]
pub enum OpenAiSseEvent {
    /// Assistant role marker (usually the first chunk).
    Role { role: String },
    /// Reasoning / think channel text, wire-order relative to content.
    Reasoning { text: String },
    /// Visible answer content.
    Content { text: String },
    /// Structured tool call fragment with stable response-scoped id + index.
    ToolCall {
        index: u32,
        id: Option<String>,
        name: Option<String>,
        /// JSON-string arguments fragment (OpenAI streaming shape).
        arguments: Option<String>,
    },
    /// Authoritative terminal choice (`finish_reason` set).
    Finish {
        reason: String,
        /// Optional timings object from the terminal chunk.
        timings: Option<Value>,
    },
    /// Separate `choices: []` usage chunk (never rides the finish choice).
    Usage { usage: Value },
    /// Literal `data: [DONE]` terminator.
    Done,
}

/// Deterministic SSE reader/parser seam over an already-open byte stream.
///
/// Successful completion requires an authoritative finish choice followed by
/// `[DONE]`. Socket EOF before that pair is [`ClientError::PrematureEof`].
/// Returning true from `cancelled` yields [`ClientError::Cancelled`] (never
/// retried client-side).
pub fn read_openai_sse(
    reader: impl BufRead,
    mut on_event: impl FnMut(OpenAiSseEvent) -> Result<()>,
    mut cancelled: impl FnMut() -> bool,
) -> Result<()> {
    let mut saw_finish = false;

    for line in reader.lines() {
        if cancelled() {
            return Err(ClientError::Cancelled);
        }
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some(payload) = trimmed.strip_prefix("data:").map(str::trim) else {
            // Ignore SSE comments / event: / id: / retry: lines.
            continue;
        };
        if payload.is_empty() {
            continue;
        }
        if payload == "[DONE]" {
            if !saw_finish {
                return Err(ClientError::PrematureEof(
                    "[DONE] without prior finish_reason choice".into(),
                ));
            }
            on_event(OpenAiSseEvent::Done)?;
            return Ok(());
        }

        let value = serde_json::from_str::<Value>(payload).map_err(|error| {
            ClientError::Http(format!("invalid SSE JSON: {error}; event={payload}"))
        })?;

        if let Some(error) = value.get("error") {
            return Err(ClientError::Http(
                error
                    .get("message")
                    .and_then(Value::as_str)
                    .or_else(|| error.as_str())
                    .unwrap_or("server error")
                    .to_owned(),
            ));
        }

        // Separate usage chunk: choices is empty array.
        if let Some(choices) = value.get("choices").and_then(Value::as_array) {
            if choices.is_empty() {
                if let Some(usage) = value.get("usage").cloned() {
                    on_event(OpenAiSseEvent::Usage { usage })?;
                }
                continue;
            }
        }

        let Some(choice) = value.get("choices").and_then(|choices| choices.get(0)) else {
            continue;
        };

        // Emit reasoning before content to preserve server field order within
        // a single delta object (server serializers put reasoning first when both
        // appear; we match that ordering contract for multi-field deltas).
        if let Some(delta) = choice.get("delta") {
            if let Some(role) = delta.get("role").and_then(Value::as_str) {
                if !role.is_empty() {
                    on_event(OpenAiSseEvent::Role {
                        role: role.to_owned(),
                    })?;
                }
            }
            if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str) {
                if !text.is_empty() {
                    on_event(OpenAiSseEvent::Reasoning {
                        text: text.to_owned(),
                    })?;
                }
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    on_event(OpenAiSseEvent::Content {
                        text: text.to_owned(),
                    })?;
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
                    let id = call
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned);
                    let name = call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned);
                    let arguments = call
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                    // Skip completely empty fragments.
                    if id.is_none()
                        && name.is_none()
                        && arguments.as_ref().is_none_or(|a| a.is_empty())
                    {
                        continue;
                    }
                    on_event(OpenAiSseEvent::ToolCall {
                        index,
                        id,
                        name,
                        arguments,
                    })?;
                }
            }
        }

        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            // OpenAI uses JSON null while streaming; only non-null strings count.
            let timings = value.get("timings").cloned();
            on_event(OpenAiSseEvent::Finish {
                reason: reason.to_owned(),
                timings,
            })?;
            saw_finish = true;
        }
    }

    if saw_finish {
        return Err(ClientError::PrematureEof(
            "finish_reason received without trailing [DONE]".into(),
        ));
    }
    Err(ClientError::PrematureEof(
        "stream ended before finish_reason and [DONE]".into(),
    ))
}

/// Stream an OpenAI-compatible chat response as typed SSE events.
///
/// The callback receives [`OpenAiSseEvent`] values in wire order. Returning
/// true from `cancelled` stops after the current SSE event with
/// [`ClientError::Cancelled`] (no client retry). Successful streams always
/// deliver a [`OpenAiSseEvent::Finish`] then [`OpenAiSseEvent::Done`].
pub fn stream_openai_chat(
    host: &str,
    port: u16,
    mut body: Value,
    timeout: Duration,
    mut on_event: impl FnMut(OpenAiSseEvent) -> Result<()>,
    mut cancelled: impl FnMut() -> bool,
) -> Result<()> {
    body["stream"] = Value::Bool(true);
    let agent = http_agent(timeout);
    let mut response = agent
        .post(&service_url(host, port, "/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .send(body.to_string())
        .map_err(|error| ClientError::Http(error.to_string()))?;
    let status = response.status().as_u16();
    if status >= 400 {
        let text = response.body_mut().read_to_string().unwrap_or_default();
        return Err(ClientError::Http(http_error_message(status, &text)));
    }

    let reader = BufReader::new(response.into_body().into_reader());
    read_openai_sse(reader, &mut on_event, &mut cancelled)
}

fn http_agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .timeout_recv_body(Some(Duration::from_secs(120)))
        .http_status_as_error(false)
        .build()
        .into()
}

fn http_error_message(status: u16, text: &str) -> String {
    let detail = serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| text.chars().take(512).collect());
    format!("HTTP {status}: {detail}")
}

pub type Result<T> = std::result::Result<T, ClientError>;

struct Dispatch {
    pending: Arc<Mutex<HashMap<(String, u64), mpsc::Sender<Value>>>>,
    control: Arc<Mutex<Option<mpsc::Sender<Value>>>>,
    // Keep handle to avoid detached thread being killed early; not joined on drop
    _handle: Option<thread::JoinHandle<()>>,
}

struct EngineInner {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    dispatch: Dispatch,
    last_state_epoch: Mutex<Option<u64>>,
    active_attempt_id: Mutex<Option<u64>>,
    last_retry_reset_eligible: Mutex<Option<bool>>,
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

impl Engine {
    pub fn spawn(daemon: impl AsRef<Path>, environment: &BTreeMap<String, String>) -> Result<Self> {
        let daemon = daemon.as_ref();
        if !daemon.is_file() {
            return Err(ClientError::MissingDaemon(daemon.to_owned()));
        }
        let mut command = Command::new(daemon);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .envs(environment);
        let mut child = command.spawn().map_err(|source| ClientError::Spawn {
            path: daemon.to_owned(),
            source,
        })?;
        let stdin = child.stdin.take().ok_or(ClientError::MissingStdin)?;
        let stdout = child.stdout.take().ok_or(ClientError::MissingStdout)?;
        let pending: Arc<Mutex<HashMap<(String, u64), mpsc::Sender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let control: Arc<Mutex<Option<mpsc::Sender<Value>>>> = Arc::new(Mutex::new(None));
        let pending_clone = Arc::clone(&pending);
        let control_clone = Arc::clone(&control);
        let handle = thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                let n = match reader.read_line(&mut line) {
                    Ok(n) => n,
                    Err(_) => break,
                };
                if n == 0 {
                    break;
                }
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let value: Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let ty = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let is_lifecycle = matches!(
                    ty,
                    "gen_start"
                        | "token"
                        | "reasoning"
                        | "tool_calls"
                        | "committed"
                        | "commit_ready"
                        | "aborted"
                        | "done"
                        | "error"
                );
                if is_lifecycle {
                    match route_lifecycle_event(ty, &value) {
                        LifecycleRoute::Pending(key) => {
                            let sender_opt = {
                                let map = pending_clone.lock().unwrap();
                                map.get(&key).cloned()
                            };
                            if let Some(sender) = sender_opt {
                                let _ = sender.send(value);
                            }
                            // Unknown/stale keyed lifecycle belongs to no live
                            // request. Never broadcast into peer channels.
                        }
                        LifecycleRoute::Control => {
                            // Id-less / invalid-key daemon errors are control-plane
                            // responses (load/config/reset/unload).
                            if let Some(sender) = control_clone.lock().unwrap().as_ref().cloned() {
                                let _ = sender.send(value);
                            }
                        }
                        LifecycleRoute::Drop => {
                            // Missing/empty/invalid keying on non-error lifecycle
                            // is quarantined — never fanned into pending channels.
                        }
                    }
                } else if let Some(sender) = control_clone.lock().unwrap().as_ref().cloned() {
                    let _ = sender.send(value);
                }
            }
            // On EOF, drop all pending senders to unblock waiters with Closed.
            {
                let mut map = pending_clone.lock().unwrap();
                map.clear();
            }
            *control_clone.lock().unwrap() = None;
        });
        let inner = EngineInner {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            dispatch: Dispatch {
                pending,
                control,
                _handle: Some(handle),
            },
            last_state_epoch: Mutex::new(None),
            active_attempt_id: Mutex::new(None),
            last_retry_reset_eligible: Mutex::new(None),
        };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Start the engine daemon and install validated process policy before the
    /// daemon initializes a GPU. This is the native control-plane path; the
    /// environment map is reserved for non-hipfire bootstrap state inherited
    /// by external launchers.
    pub fn spawn_configured(
        daemon: impl AsRef<Path>,
        environment: &BTreeMap<String, String>,
        config: &hipfire_config::ProcessConfig,
    ) -> Result<Self> {
        let mut environment = environment.clone();
        let hip = environment
            .get(hipfire_config::HIP_VISIBLE_DEVICES)
            .cloned()
            .or_else(|| std::env::var(hipfire_config::HIP_VISIBLE_DEVICES).ok());
        let rocr = environment
            .get(hipfire_config::ROCR_VISIBLE_DEVICES)
            .cloned()
            .or_else(|| std::env::var(hipfire_config::ROCR_VISIBLE_DEVICES).ok());
        let visibility =
            hipfire_config::synchronized_device_visibility(config, hip.as_deref(), rocr.as_deref())
                .map_err(|error| ClientError::Protocol(error.to_string()))?;
        if let Some(visibility) = visibility {
            environment.insert(hipfire_config::HIP_VISIBLE_DEVICES.into(), visibility.hip);
            environment.insert(hipfire_config::ROCR_VISIBLE_DEVICES.into(), visibility.rocr);
        }

        let engine = Self::spawn(daemon, &environment)?;
        let response = engine.request(&serde_json::json!({
            "type": "configure",
            "config": config,
        }))?;
        expect_type(&response, "configured")?;
        Ok(engine)
    }

    pub fn send(&self, message: &Value) -> Result<()> {
        let mut stdin = self.inner.stdin.lock().unwrap();
        serde_json::to_writer(&mut *stdin, message).map_err(|error| {
            ClientError::Protocol(format!("failed to serialize request: {error}"))
        })?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        Ok(())
    }

    fn recv_control(&self, rx: &mpsc::Receiver<Value>) -> Result<Value> {
        rx.recv().map_err(|_| {
            let status = self
                .inner
                .child
                .lock()
                .unwrap()
                .try_wait()
                .ok()
                .flatten()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "unknown".into());
            ClientError::Closed { status }
        })
    }

    pub fn request(&self, message: &Value) -> Result<Value> {
        let (tx, rx) = mpsc::channel();
        *self.inner.dispatch.control.lock().unwrap() = Some(tx);
        let send_res = self.send(message);
        if let Err(e) = send_res {
            *self.inner.dispatch.control.lock().unwrap() = None;
            return Err(e);
        }
        let value = self.recv_control(&rx);
        *self.inner.dispatch.control.lock().unwrap() = None;
        value
    }

    /// Compatibility recv for control plane (loads use loop). Exposed for tests that need direct control recv.
    pub fn recv(&self) -> Result<Value> {
        // For backward compat, create a one-shot control waiter and block.
        // This is used only in tests that call recv directly after send; for multiplexed
        // generate paths we use per-request channels.
        let (tx, rx) = mpsc::channel();
        *self.inner.dispatch.control.lock().unwrap() = Some(tx);
        let v = self.recv_control(&rx);
        *self.inner.dispatch.control.lock().unwrap() = None;
        v
    }

    pub fn ping(&self) -> Result<()> {
        let response = self.request(&serde_json::json!({ "type": "ping" }))?;
        expect_type(&response, "pong")
    }

    pub fn load(&self, model: &Path, params: Value) -> Result<Value> {
        // Drop prior eligibility before load so a malformed/omitted field cannot
        // leave a stale true from a previous model.
        *self.inner.last_retry_reset_eligible.lock().unwrap() = None;
        let (tx, rx) = mpsc::channel();
        *self.inner.dispatch.control.lock().unwrap() = Some(tx);
        let send_res = self.send(&serde_json::json!({
            "type": "load",
            "model": model,
            "params": params,
        }));
        if let Err(e) = send_res {
            *self.inner.dispatch.control.lock().unwrap() = None;
            return Err(e);
        }
        let mut response = match self.recv_control(&rx) {
            Ok(v) => v,
            Err(e) => {
                *self.inner.dispatch.control.lock().unwrap() = None;
                return Err(e);
            }
        };
        if response.get("type").and_then(Value::as_str) == Some("error") {
            *self.inner.dispatch.control.lock().unwrap() = None;
            return Err(daemon_error_from_value(&response));
        }
        while response.get("type").and_then(Value::as_str) != Some("loaded") {
            response = match self.recv_control(&rx) {
                Ok(v) => v,
                Err(e) => {
                    *self.inner.dispatch.control.lock().unwrap() = None;
                    return Err(e);
                }
            };
            if response.get("type").and_then(Value::as_str) == Some("error") {
                *self.inner.dispatch.control.lock().unwrap() = None;
                return Err(daemon_error_from_value(&response));
            }
        }
        let eligible = match validate_retry_reset_eligible_field(&response) {
            Ok(e) => e,
            Err(err) => {
                *self.inner.dispatch.control.lock().unwrap() = None;
                return Err(err);
            }
        };
        *self.inner.last_retry_reset_eligible.lock().unwrap() = Some(eligible);
        *self.inner.dispatch.control.lock().unwrap() = None;
        Ok(response)
    }

    pub fn generate(
        &self,
        request: &Value,
        mut event: impl FnMut(&Value) -> Result<()>,
    ) -> Result<Value> {
        let request_id = request
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ClientError::Protocol("generate request missing id".into()))?
            .to_owned();
        let attempt_id = require_attempt_id(request.get("attempt_id"))
            .map_err(|reason| ClientError::Protocol(format!("generate request {reason}")))?;
        let (tx, rx) = mpsc::channel();
        {
            let mut map = self.inner.dispatch.pending.lock().unwrap();
            let key = (request_id.clone(), attempt_id);
            if map.contains_key(&key) {
                // Leave the original live sender and active-attempt bookkeeping
                // undisturbed; never send a duplicate generate request.
                return Err(ClientError::Protocol(format!(
                    "duplicate live generate id={request_id} attempt_id={attempt_id}"
                )));
            }
            map.insert(key, tx);
        }
        *self.inner.active_attempt_id.lock().unwrap() = Some(attempt_id);
        let send_res = self.send(request);
        if let Err(err) = send_res {
            self.inner
                .dispatch
                .pending
                .lock()
                .unwrap()
                .remove(&(request_id.clone(), attempt_id));
            *self.inner.active_attempt_id.lock().unwrap() = None;
            return Err(err);
        }
        let res = self.generate_with_rx(&request_id, attempt_id, &rx, &mut event);
        self.inner
            .dispatch
            .pending
            .lock()
            .unwrap()
            .remove(&(request_id, attempt_id));
        *self.inner.active_attempt_id.lock().unwrap() = None;
        res
    }

    fn generate_with_rx(
        &self,
        request_id: &str,
        attempt_id: u64,
        rx: &mpsc::Receiver<Value>,
        event: &mut dyn FnMut(&Value) -> Result<()>,
    ) -> Result<Value> {
        loop {
            let value = match self.recv_control(rx) {
                Ok(v) => v,
                Err(err) => {
                    return Err(err);
                }
            };
            if let Some(err) = reject_stale_lifecycle_event(&value, request_id, Some(attempt_id)) {
                return Err(err);
            }

            let ty = value.get("type").and_then(Value::as_str);
            if ty == Some("commit_ready") {
                match event(&value) {
                    Ok(()) => {
                        let commit = serde_json::json!({
                            "type": "commit",
                            "id": request_id,
                            "attempt_id": attempt_id,
                        });
                        if let Err(err) = self.send(&commit) {
                            return self.abort_and_drain_with_rx(request_id, attempt_id, rx, err);
                        }
                        return self
                            .drain_committed_done_with_rx(request_id, attempt_id, rx, &value);
                    }
                    Err(cb_err) => {
                        return self.abort_and_drain_with_rx(request_id, attempt_id, rx, cb_err);
                    }
                }
            }

            if let Err(cb_err) = event(&value) {
                // A terminal event has already closed the transaction daemon-side, so
                // `apply_terminal_control` drops the abort and never emits the
                // `aborted`+`done` pair this drain blocks on. Same rule the post-commit
                // path already follows.
                if matches!(ty, Some("error") | Some("done")) {
                    return Err(cb_err);
                }
                return self.abort_and_drain_with_rx(request_id, attempt_id, rx, cb_err);
            }

            match ty {
                Some("done") => {
                    return Ok(value);
                }
                Some("error") => {
                    return Err(daemon_error_from_value(&value));
                }
                _ => {}
            }
        }
    }

    fn drain_committed_done_with_rx(
        &self,
        request_id: &str,
        attempt_id: u64,
        rx: &mpsc::Receiver<Value>,
        staged_commit_ready: &Value,
    ) -> Result<Value> {
        // Reads exactly one control event and dispatches on it; every arm is
        // terminal, so this is deliberately not a loop.
        let value = self.recv_control(rx)?;
        if let Some(err) = reject_stale_lifecycle_event(&value, request_id, Some(attempt_id)) {
            return Err(err);
        }
        match value.get("type").and_then(Value::as_str) {
            Some("done") => {
                let finish = value.get("finish_reason").and_then(Value::as_str);
                if finish == Some("aborted") {
                    return Err(ClientError::Protocol(
                        "committed drain rejected aborted done after commit".into(),
                    ));
                }
                if let Some(err) = committed_done_payload_mismatch(staged_commit_ready, &value) {
                    return Err(err);
                }
                Ok(value)
            }
            Some("error") => Err(daemon_error_from_value(&value)),
            Some(other) => Err(ClientError::Protocol(format!(
                "committed drain unexpected event type={other}"
            ))),
            None => Err(ClientError::Protocol(
                "committed drain event missing type".into(),
            )),
        }
    }

    fn abort_and_drain_with_rx(
        &self,
        request_id: &str,
        attempt_id: u64,
        rx: &mpsc::Receiver<Value>,
        original: ClientError,
    ) -> Result<Value> {
        let abort = serde_json::json!({
            "type": "abort",
            "id": request_id,
            "attempt_id": attempt_id,
        });
        if let Err(err) = self.send(&abort) {
            return Err(err);
        }

        let mut saw_aborted = false;
        loop {
            let value = match self.recv_control(rx) {
                Ok(v) => v,
                Err(err) => {
                    return Err(err);
                }
            };
            // Missing/malformed lifecycle keys are dropped by the dispatcher and
            // never delivered to this request channel. Unknown-but-well-formed
            // keys can still arrive if the daemon emitted one before this request
            // registered; preserve cancellation rather than poisoning the drain.
            if value.get("id").and_then(Value::as_str) != Some(request_id)
                || parse_attempt_id(value.get("attempt_id")) != Some(attempt_id)
            {
                continue;
            }
            match value.get("type").and_then(Value::as_str) {
                Some("aborted") => {
                    saw_aborted = true;
                }
                Some("done") => {
                    let finish = value.get("finish_reason").and_then(Value::as_str);
                    if finish == Some("aborted") {
                        // Malformed earlier records were quarantined by the
                        // dispatcher. Once the daemon's own well-formed aborted
                        // done arrives, the client cancellation is complete even
                        // if no valid `aborted` marker survived.
                        return Err(original);
                    }
                    return Err(ClientError::Protocol(format!(
                        "cancellation drain rejected non-aborted done \
                         (saw_aborted={saw_aborted}, finish_reason={finish:?})"
                    )));
                }
                Some("error") => {
                    return Err(daemon_error_from_value(&value));
                }
                _ => {}
            }
        }
    }

    /// Cold-reset the loaded model, correlating the failed/upcoming attempt.
    ///
    /// `attempt_id` is mandatory on the wire and must be echoed exactly on the
    /// reset acknowledgement. Task 7 owns the sole production caller and will
    /// pass the allocated attempt id.
    pub fn reset(&self, attempt_id: u64) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        *self.inner.dispatch.control.lock().unwrap() = Some(tx);
        let send_res = self.send(&serde_json::json!({
            "type": "reset",
            "attempt_id": attempt_id,
        }));
        if let Err(e) = send_res {
            *self.inner.dispatch.control.lock().unwrap() = None;
            return Err(e);
        }
        let response = match self.recv_control(&rx) {
            Ok(v) => v,
            Err(e) => {
                *self.inner.dispatch.control.lock().unwrap() = None;
                return Err(e);
            }
        };
        *self.inner.dispatch.control.lock().unwrap() = None;
        self.validate_reset_ack(&response, attempt_id)
    }

    /// Last successful cold-reset epoch observed on this connection.
    pub fn last_state_epoch(&self) -> Option<u64> {
        *self.inner.last_state_epoch.lock().unwrap()
    }

    /// Current active attempt id if a generate is in-flight.
    pub fn active_attempt_id(&self) -> Option<u64> {
        *self.inner.active_attempt_id.lock().unwrap()
    }

    /// Last daemon-attested `retry_reset_eligible` flag (reset ack or loaded).
    pub fn last_retry_reset_eligible(&self) -> Option<bool> {
        *self.inner.last_retry_reset_eligible.lock().unwrap()
    }

    /// Record eligibility from a capability response (`loaded` / reset ack).
    pub fn note_retry_reset_eligible(&self, eligible: bool) {
        *self.inner.last_retry_reset_eligible.lock().unwrap() = Some(eligible);
    }

    fn validate_reset_ack(&self, response: &Value, expected_attempt: u64) -> Result<()> {
        if response.get("type").and_then(Value::as_str) == Some("error") {
            return Err(daemon_error_from_value(response));
        }
        if response.get("type").and_then(Value::as_str) != Some("reset") {
            return Err(ClientError::Protocol(format!(
                "expected reset, received {}",
                response
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("missing type")
            )));
        }
        let rolled_back = response.get("rolled_back").and_then(Value::as_bool);
        if rolled_back != Some(true) {
            return Err(ClientError::Protocol(
                "reset ack missing rolled_back=true".into(),
            ));
        }
        let seq_pos = response.get("seq_pos").and_then(Value::as_u64);
        if seq_pos != Some(0) {
            return Err(ClientError::Protocol(format!(
                "reset ack requires seq_pos=0, got {seq_pos:?}"
            )));
        }
        let conversation_len = response.get("conversation_len").and_then(Value::as_u64);
        if conversation_len != Some(0) {
            return Err(ClientError::Protocol(format!(
                "reset ack requires conversation_len=0, got {conversation_len:?}"
            )));
        }
        let attempt_id = require_attempt_id(response.get("attempt_id"))
            .map_err(|reason| ClientError::Protocol(format!("reset ack {reason}")))?;
        if attempt_id != expected_attempt {
            return Err(ClientError::Protocol(format!(
                "reset ack attempt_id mismatch: expected {expected_attempt}, got {attempt_id}"
            )));
        }
        let epoch = response
            .get("state_epoch")
            .and_then(Value::as_u64)
            .ok_or_else(|| ClientError::Protocol("reset ack missing state_epoch".into()))?;
        if let Some(prev) = *self.inner.last_state_epoch.lock().unwrap() {
            if epoch <= prev {
                return Err(ClientError::Protocol(format!(
                    "reset ack state_epoch not strictly increasing: prev={prev} got={epoch}"
                )));
            }
        }
        let eligible = response
            .get("retry_reset_eligible")
            .and_then(Value::as_bool)
            .ok_or_else(|| {
                ClientError::Protocol("reset ack missing retry_reset_eligible".into())
            })?;
        {
            let mut guard = self.inner.last_state_epoch.lock().unwrap();
            // Re-check under lock to avoid race after the initial check.
            if let Some(prev) = *guard {
                if epoch <= prev {
                    return Err(ClientError::Protocol(format!(
                        "reset ack state_epoch not strictly increasing: prev={prev} got={epoch}"
                    )));
                }
            }
            *guard = Some(epoch);
        }
        *self.inner.last_retry_reset_eligible.lock().unwrap() = Some(eligible);
        Ok(())
    }

    pub fn unload(&self) -> Result<()> {
        let response = self.request(&serde_json::json!({ "type": "unload" }))?;
        expect_type(&response, "unloaded")
    }

    pub fn child_id(&self) -> u32 {
        self.inner.child.lock().unwrap().id()
    }
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        // EngineInner is dropped exactly once after every Engine clone is gone.
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
        // Dispatch thread will exit on EOF; we don't join to avoid blocking drop.
    }
}

fn expect_type(response: &Value, expected: &str) -> Result<()> {
    match response.get("type").and_then(Value::as_str) {
        Some(actual) if actual == expected => Ok(()),
        Some("error") => Err(daemon_error_from_value(response)),
        actual => Err(ClientError::Protocol(format!(
            "expected {expected}, received {}",
            actual.unwrap_or("missing type")
        ))),
    }
}

fn error_message(value: &Value) -> String {
    value
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("unknown daemon error")
        .to_owned()
}
/// Parse a numeric attempt id. Accepts JSON numbers only (u64 or non-neg i64).
/// Decimal strings are rejected — no further coercion.
fn parse_attempt_id(value: Option<&Value>) -> Option<u64> {
    let value = value?;
    if let Some(n) = value.as_u64() {
        return Some(n);
    }
    if let Some(n) = value.as_i64() {
        if n >= 0 {
            return Some(n as u64);
        }
    }
    None
}

/// Where a generation-lifecycle JSONL record should be delivered.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LifecycleRoute {
    /// Exact live generate channel for `(id, attempt_id)`.
    Pending((String, u64)),
    /// Active control-plane waiter (load/config/reset/unload).
    Control,
    /// Quarantine: never fan out into peer generation channels.
    Drop,
}

/// Route a lifecycle event by its generation key.
///
/// Valid nonempty `(id, attempt_id)` always targets that pending key (including
/// generation errors). An `error` lacking a valid key is a control-plane
/// response. Every other unkeyed/malformed lifecycle record is dropped.
fn route_lifecycle_event(ty: &str, value: &Value) -> LifecycleRoute {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty());
    let attempt = parse_attempt_id(value.get("attempt_id"));
    match (id, attempt) {
        (Some(id), Some(attempt)) => LifecycleRoute::Pending((id.to_owned(), attempt)),
        _ if ty == "error" => LifecycleRoute::Control,
        _ => LifecycleRoute::Drop,
    }
}

/// Require a present, numeric attempt_id. Distinguishes missing vs malformed.
fn require_attempt_id(value: Option<&Value>) -> std::result::Result<u64, &'static str> {
    match value {
        None => Err("missing attempt_id"),
        Some(v) => parse_attempt_id(Some(v)).ok_or("malformed attempt_id"),
    }
}

/// Decode a daemon `type=error` event into [`ClientError::Daemon`] when the
/// typed envelope is present; otherwise fall back to protocol/message-only.
pub fn daemon_error_from_value(value: &Value) -> ClientError {
    let message = error_message(value);
    let class = value
        .get("class")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let retryable = value.get("retryable").and_then(Value::as_bool);
    let rolled_back = value.get("rolled_back").and_then(Value::as_bool);
    let attempt_id = parse_attempt_id(value.get("attempt_id"));
    match (class, retryable, rolled_back, attempt_id) {
        (Some(class), Some(retryable), Some(rolled_back), Some(attempt_id)) => {
            ClientError::Daemon(TypedDaemonError {
                message,
                class,
                retryable,
                rolled_back,
                attempt_id,
                id: value.get("id").and_then(Value::as_str).map(str::to_owned),
            })
        }
        _ => ClientError::Protocol(message),
    }
}

/// Reject generation lifecycle events with missing, malformed, mismatched, or
/// stale `id` / `attempt_id` relative to the active generate request.
///
/// Every lifecycle event must carry the exact nonempty request id and the exact
/// numeric attempt id. Wrong-id events from another request are rejected before
/// callbacks or terminal acceptance.
fn reject_stale_lifecycle_event(
    value: &Value,
    request_id: &str,
    active: Option<u64>,
) -> Option<ClientError> {
    let Some(active) = active else {
        return None;
    };
    let ty = value.get("type").and_then(Value::as_str)?;
    let is_lifecycle = matches!(
        ty,
        "gen_start"
            | "token"
            | "reasoning"
            | "tool_calls"
            | "committed"
            | "commit_ready"
            | "aborted"
            | "done"
            | "error"
    );
    if !is_lifecycle {
        return None;
    }
    match value.get("id").and_then(Value::as_str) {
        None => {
            return Some(ClientError::Protocol(format!(
                "missing id on {ty} event (expected={request_id})"
            )));
        }
        Some(id) if id.is_empty() => {
            return Some(ClientError::Protocol(format!(
                "empty id on {ty} event (expected={request_id})"
            )));
        }
        Some(id) if id != request_id => {
            return Some(ClientError::Protocol(format!(
                "stale id event: expected={request_id} got={id} type={ty}"
            )));
        }
        Some(_) => {}
    }
    match value.get("attempt_id") {
        None => Some(ClientError::Protocol(format!(
            "missing attempt_id on {ty} event (active={active})"
        ))),
        Some(raw) => match parse_attempt_id(Some(raw)) {
            None => Some(ClientError::Protocol(format!(
                "malformed attempt_id on {ty} event (active={active})"
            ))),
            Some(got) if got != active => Some(ClientError::Protocol(format!(
                "stale attempt event: active={active} got={got} type={ty}"
            ))),
            Some(_) => None,
        },
    }
}

/// Compare final `done` to staged `commit_ready` after normalizing only `type`.
/// Any other payload divergence is a protocol error.
fn committed_done_payload_mismatch(
    staged_commit_ready: &Value,
    done: &Value,
) -> Option<ClientError> {
    let mut staged = staged_commit_ready.clone();
    let mut final_done = done.clone();
    if let Some(obj) = staged.as_object_mut() {
        obj.insert("type".into(), Value::String("done".into()));
    }
    if let Some(obj) = final_done.as_object_mut() {
        obj.insert("type".into(), Value::String("done".into()));
    }
    if staged == final_done {
        None
    } else {
        Some(ClientError::Protocol(
            "committed done payload diverges from staged commit_ready".into(),
        ))
    }
}

/// Validate a daemon-attested `retry_reset_eligible` capability bit.
///
/// Policy lives in `hipfire_runtime::reset_core` and is echoed on reset ack /
/// loaded responses. The client never maintains an independent arch allowlist.
pub fn validate_retry_reset_eligible_field(value: &Value) -> Result<bool> {
    value
        .get("retry_reset_eligible")
        .and_then(Value::as_bool)
        .ok_or_else(|| ClientError::Protocol("missing or non-bool retry_reset_eligible".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::{env, fs};

    fn collect_sse(raw: &str) -> Result<Vec<OpenAiSseEvent>> {
        let mut out = Vec::new();
        read_openai_sse(
            Cursor::new(raw.as_bytes()),
            |ev| {
                out.push(ev);
                Ok(())
            },
            || false,
        )?;
        Ok(out)
    }

    fn sse_line(json: &str) -> String {
        format!("data: {json}\n\n")
    }

    #[test]
    fn sse_reasoning_then_content_order_preserved() {
        let raw = format!(
            "{}{}{}{}",
            sse_line(
                r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#
            ),
            sse_line(
                r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}"#
            ),
            sse_line(
                r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#
            ),
            sse_line(
                r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"timings":{"decode_tok_s":10.0}}"#
            ),
        ) + "data: [DONE]\n\n";
        let events = collect_sse(&raw).expect("ok stream");
        assert_eq!(
            events,
            vec![
                OpenAiSseEvent::Role {
                    role: "assistant".into()
                },
                OpenAiSseEvent::Reasoning {
                    text: "think".into()
                },
                OpenAiSseEvent::Content { text: "hi".into() },
                OpenAiSseEvent::Finish {
                    reason: "stop".into(),
                    timings: Some(serde_json::json!({"decode_tok_s": 10.0})),
                },
                OpenAiSseEvent::Done,
            ]
        );
    }

    #[test]
    fn sse_same_delta_reasoning_before_content() {
        // Within one delta object, reasoning is emitted before content.
        let raw = format!(
            "{}{}",
            sse_line(
                r#"{"choices":[{"index":0,"delta":{"reasoning_content":"r","content":"c"},"finish_reason":null}]}"#
            ),
            sse_line(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
        ) + "data: [DONE]\n\n";
        let events = collect_sse(&raw).unwrap();
        let kinds: Vec<&str> = events
            .iter()
            .map(|e| match e {
                OpenAiSseEvent::Reasoning { .. } => "r",
                OpenAiSseEvent::Content { .. } => "c",
                OpenAiSseEvent::Finish { .. } => "f",
                OpenAiSseEvent::Done => "d",
                _ => "?",
            })
            .collect();
        assert_eq!(kinds, vec!["r", "c", "f", "d"]);
    }

    #[test]
    fn sse_tool_calls_stable_ids_and_indices() {
        let raw = format!(
            "{}{}{}",
            sse_line(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}},{"index":1,"id":"call_1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"b.rs\"}"}}]},"finish_reason":null}]}"#
            ),
            sse_line(
                r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"timings":{"ttft_ms":1}}"#
            ),
            sse_line(
                r#"{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}"#
            ),
        ) + "data: [DONE]\n\n";
        let events = collect_sse(&raw).unwrap();
        assert_eq!(
            events[0],
            OpenAiSseEvent::ToolCall {
                index: 0,
                id: Some("call_0".into()),
                name: Some("read_file".into()),
                arguments: Some(r#"{"path":"a.rs"}"#.into()),
            }
        );
        assert_eq!(
            events[1],
            OpenAiSseEvent::ToolCall {
                index: 1,
                id: Some("call_1".into()),
                name: Some("write_file".into()),
                arguments: Some(r#"{"path":"b.rs"}"#.into()),
            }
        );
        match &events[2] {
            OpenAiSseEvent::Finish { reason, timings } => {
                assert_eq!(reason, "tool_calls");
                assert!(timings.is_some());
            }
            other => panic!("expected Finish, got {other:?}"),
        }
        match &events[3] {
            OpenAiSseEvent::Usage { usage } => {
                assert_eq!(usage["prompt_tokens"], 3);
                assert_eq!(usage["completion_tokens"], 5);
            }
            other => panic!("expected Usage, got {other:?}"),
        }
        assert_eq!(events[4], OpenAiSseEvent::Done);
    }

    #[test]
    fn sse_finish_usage_then_done_success() {
        let raw = format!(
            "{}{}{}",
            sse_line(r#"{"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}"#),
            sse_line(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
            sse_line(
                r#"{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#
            ),
        ) + "data: [DONE]\n\n";
        let events = collect_sse(&raw).unwrap();
        assert!(matches!(events[0], OpenAiSseEvent::Content { .. }));
        assert!(matches!(
            events[1],
            OpenAiSseEvent::Finish { reason: _, .. }
        ));
        assert!(matches!(events[2], OpenAiSseEvent::Usage { .. }));
        assert_eq!(events.last(), Some(&OpenAiSseEvent::Done));
    }

    #[test]
    fn sse_server_error_event_is_error() {
        let raw = sse_line(r#"{"error":{"message":"boom","type":"server_error"}}"#);
        let err = collect_sse(&raw).unwrap_err();
        assert!(
            matches!(&err, ClientError::Http(m) if m.contains("boom")),
            "{err}"
        );
    }

    #[test]
    fn sse_invalid_json_is_error() {
        let raw = "data: {not-json\n\n";
        let err = collect_sse(raw).unwrap_err();
        assert!(
            matches!(&err, ClientError::Http(m) if m.contains("invalid SSE JSON")),
            "{err}"
        );
    }

    #[test]
    fn sse_done_without_finish_is_premature() {
        let raw = "data: [DONE]\n\n";
        let err = collect_sse(raw).unwrap_err();
        assert!(
            matches!(&err, ClientError::PrematureEof(m) if m.contains("without prior finish")),
            "{err}"
        );
    }

    #[test]
    fn sse_finish_without_done_is_premature() {
        let raw = sse_line(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#);
        let err = collect_sse(&raw).unwrap_err();
        assert!(
            matches!(&err, ClientError::PrematureEof(m) if m.contains("without trailing [DONE]")),
            "{err}"
        );
    }

    #[test]
    fn sse_premature_eof_mid_stream() {
        let raw = sse_line(
            r#"{"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}"#,
        );
        let err = collect_sse(&raw).unwrap_err();
        assert!(
            matches!(&err, ClientError::PrematureEof(m) if m.contains("before finish_reason")),
            "{err}"
        );
    }

    #[test]
    fn sse_empty_body_is_premature_eof() {
        let err = collect_sse("").unwrap_err();
        assert!(matches!(err, ClientError::PrematureEof(_)), "{err}");
    }

    #[test]
    fn sse_explicit_cancel_returns_cancelled() {
        let raw = format!(
            "{}{}",
            sse_line(r#"{"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}"#),
            sse_line(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
        ) + "data: [DONE]\n\n";
        let mut n = 0usize;
        let err = read_openai_sse(
            Cursor::new(raw.as_bytes()),
            |_| Ok(()),
            || {
                n += 1;
                n > 1
            },
        )
        .unwrap_err();
        assert!(matches!(err, ClientError::Cancelled), "{err}");
    }

    #[test]
    fn sse_length_finish_no_tool_calls() {
        let raw = format!(
            "{}{}",
            sse_line(
                r#"{"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}"#
            ),
            sse_line(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#),
        ) + "data: [DONE]\n\n";
        let events = collect_sse(&raw).unwrap();
        assert!(events
            .iter()
            .all(|e| !matches!(e, OpenAiSseEvent::ToolCall { .. })));
        assert!(matches!(
            events
                .iter()
                .find(|e| matches!(e, OpenAiSseEvent::Finish { .. })),
            Some(OpenAiSseEvent::Finish { reason, .. }) if reason == "length"
        ));
        assert_eq!(events.last(), Some(&OpenAiSseEvent::Done));
    }

    #[test]
    fn sse_ignores_non_data_lines() {
        let raw = format!(
            "event: message\n: keep-alive\nid: 1\n{}{}",
            sse_line(r#"{"choices":[{"index":0,"delta":{"content":"z"},"finish_reason":null}]}"#),
            sse_line(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
        ) + "data: [DONE]\n\n";
        let events = collect_sse(&raw).unwrap();
        assert!(matches!(&events[0], OpenAiSseEvent::Content { text: t } if t == "z"));
        assert_eq!(events.last(), Some(&OpenAiSseEvent::Done));
    }

    #[test]
    fn missing_daemon_fails_before_spawn() {
        let error = Engine::spawn("/definitely/missing/hipfire-daemon", &BTreeMap::new())
            .err()
            .expect("missing path fails");
        assert!(matches!(error, ClientError::MissingDaemon(_)));
    }

    #[cfg(unix)]
    #[test]
    fn jsonl_transport_frames_ping_and_unload() {
        let root = env::temp_dir().join(format!("hipfire-client-test-{}", std::process::id()));
        let daemon = write_fake_daemon(
            &root,
            "#!/bin/sh\nwhile IFS= read -r line; do\n case \"$line\" in *'\"ping\"'*) echo '{\"type\":\"pong\"}' ;; *'\"unload\"'*) echo '{\"type\":\"unloaded\"}'; exit 0 ;; esac\ndone\n",
        );
        let mut engine = spawn_fake_engine(&daemon);
        engine.ping().unwrap();
        engine.unload().unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn configured_spawn_sends_process_policy_before_ping() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-configured-test-{}",
            std::process::id()
        ));
        let daemon = write_fake_daemon(
            &root,
            "#!/bin/sh\nconfigured=0\nwhile IFS= read -r line; do\n case \"$line\" in *'\"configure\"'*) if [ \"$HIP_VISIBLE_DEVICES\" = '0,1' ] && [ \"$ROCR_VISIBLE_DEVICES\" = '2,3' ]; then configured=1; echo '{\"type\":\"configured\"}'; else echo '{\"type\":\"error\",\"message\":\"device visibility is not synchronized\"}'; fi ;; *'\"ping\"'*) if [ \"$configured\" = 1 ]; then echo '{\"type\":\"pong\"}'; else echo '{\"type\":\"error\",\"message\":\"not configured\"}'; fi ;; *'\"unload\"'*) echo '{\"type\":\"unloaded\"}'; exit 0 ;; esac\ndone\n",
        );
        let mut layer = hipfire_config::ConfigLayer::default();
        layer.set_cli("hardware.devices", "2,3").unwrap();
        let resolved = hipfire_config::resolve([hipfire_config::NamedLayer {
            source: hipfire_config::ConfigSource::GlobalUser {
                path: root.join("config.toml"),
            },
            layer,
        }])
        .unwrap();
        let config = hipfire_config::ProcessConfig::from_resolved(&resolved).unwrap();
        let mut engine = spawn_fake_engine_configured(&daemon, &config);
        engine.ping().unwrap();
        engine.unload().unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn typed_error_decode_distinguishes_retryable_fields() {
        let transient = serde_json::json!({
            "type": "error",
            "id": "r1",
            "message": "gpu hiccup",
            "class": error_class::TRANSIENT,
            "retryable": true,
            "rolled_back": true,
            "attempt_id": 3,
        });
        let err = daemon_error_from_value(&transient);
        let typed = err.typed_daemon().expect("typed");
        assert_eq!(typed.class, error_class::TRANSIENT);
        assert!(typed.retryable);
        assert!(typed.rolled_back);
        assert_eq!(typed.attempt_id, 3);

        let malformed = serde_json::json!({
            "type": "error",
            "message": "bad tool json",
            "class": error_class::MALFORMED,
            "retryable": false,
            "rolled_back": false,
            "attempt_id": 1,
        });
        let err = daemon_error_from_value(&malformed);
        let typed = err.typed_daemon().expect("typed");
        assert_eq!(typed.class, error_class::MALFORMED);
        assert!(!typed.retryable);
        assert!(!typed.rolled_back);

        // Missing typed fields → protocol string only (no false retryable).
        let legacy = serde_json::json!({"type":"error","message":"legacy"});
        assert!(matches!(
            daemon_error_from_value(&legacy),
            ClientError::Protocol(_)
        ));
    }

    fn good_reset_ack(attempt: u64, epoch: u64, eligible: bool) -> Value {
        serde_json::json!({
            "type": "reset",
            "rolled_back": true,
            "state_epoch": epoch,
            "seq_pos": 0,
            "conversation_len": 0,
            "attempt_id": attempt,
            "retry_reset_eligible": eligible,
        })
    }

    fn dummy_engine() -> Engine {
        let child = dummy_child();
        let stdin = dummy_stdin();
        let stdout = dummy_stdout();
        let pending = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        let control = std::sync::Arc::new(std::sync::Mutex::new(None));
        let _ = stdout;
        let inner = EngineInner {
            child: std::sync::Mutex::new(child),
            stdin: std::sync::Mutex::new(stdin),
            dispatch: Dispatch {
                pending,
                control,
                _handle: None,
            },
            last_state_epoch: std::sync::Mutex::new(None),
            active_attempt_id: std::sync::Mutex::new(None),
            last_retry_reset_eligible: std::sync::Mutex::new(None),
        };
        Engine {
            inner: std::sync::Arc::new(inner),
        }
    }

    #[test]
    fn reset_ack_rejects_missing_and_bad_fields() {
        let mut engine = dummy_engine();

        // type-only (legacy) must fail.
        let err = engine
            .validate_reset_ack(&serde_json::json!({"type":"reset","seq_pos":0}), 1)
            .unwrap_err();
        assert!(matches!(err, ClientError::Protocol(_)), "{err}");
        // nonzero seq_pos
        let err = engine
            .validate_reset_ack(
                &serde_json::json!({
                    "type":"reset","rolled_back":true,"state_epoch":1,
                    "seq_pos":1,"conversation_len":0,"attempt_id":1,
                    "retry_reset_eligible":true
                }),
                1,
            )
            .unwrap_err();
        assert!(err.to_string().contains("seq_pos"), "{err}");

        // nonzero conversation_len
        let err = engine
            .validate_reset_ack(
                &serde_json::json!({
                    "type":"reset","rolled_back":true,"state_epoch":1,
                    "seq_pos":0,"conversation_len":2,"attempt_id":1,
                    "retry_reset_eligible":true
                }),
                1,
            )
            .unwrap_err();
        assert!(err.to_string().contains("conversation_len"), "{err}");
        // missing attempt_id
        let err = engine
            .validate_reset_ack(
                &serde_json::json!({
                    "type":"reset","rolled_back":true,"state_epoch":1,
                    "seq_pos":0,"conversation_len":0,
                    "retry_reset_eligible":true
                }),
                1,
            )
            .unwrap_err();
        assert!(err.to_string().contains("attempt_id"), "{err}");

        // malformed attempt_id (string)
        let err = engine
            .validate_reset_ack(
                &serde_json::json!({
                    "type":"reset","rolled_back":true,"state_epoch":1,
                    "seq_pos":0,"conversation_len":0,"attempt_id":"7",
                    "retry_reset_eligible":true
                }),
                7,
            )
            .unwrap_err();
        assert!(err.to_string().contains("malformed attempt_id"), "{err}");

        // missing retry_reset_eligible
        let err = engine
            .validate_reset_ack(
                &serde_json::json!({
                    "type":"reset","rolled_back":true,"state_epoch":1,
                    "seq_pos":0,"conversation_len":0,"attempt_id":7
                }),
                7,
            )
            .unwrap_err();
        assert!(err.to_string().contains("retry_reset_eligible"), "{err}");

        // rolled_back false
        let err = engine
            .validate_reset_ack(
                &serde_json::json!({
                    "type":"reset","rolled_back":false,"state_epoch":1,
                    "seq_pos":0,"conversation_len":0,"attempt_id":1,
                    "retry_reset_eligible":true
                }),
                1,
            )
            .unwrap_err();
        assert!(err.to_string().contains("rolled_back"), "{err}");

        // good ack — equality always required
        engine
            .validate_reset_ack(&good_reset_ack(7, 1, true), 7)
            .unwrap();
        assert_eq!(engine.last_state_epoch(), Some(1));
        assert_eq!(engine.last_retry_reset_eligible(), Some(true));

        // non-increasing epoch
        let err = engine
            .validate_reset_ack(&good_reset_ack(8, 1, false), 8)
            .unwrap_err();
        assert!(err.to_string().contains("state_epoch"), "{err}");

        // attempt mismatch
        let err = engine
            .validate_reset_ack(&good_reset_ack(9, 2, true), 8)
            .unwrap_err();
        assert!(err.to_string().contains("attempt_id mismatch"), "{err}");

        // increasing epoch ok
        engine
            .validate_reset_ack(&good_reset_ack(8, 2, false), 8)
            .unwrap();
        assert_eq!(engine.last_state_epoch(), Some(2));
        assert_eq!(engine.last_retry_reset_eligible(), Some(false));
    }

    #[test]
    fn lifecycle_id_and_attempt_required_and_matched() {
        let active = Some(2u64);
        let rid = "r";

        let stale = serde_json::json!({"type":"token","id":"r","text":"x","attempt_id":1});
        let err = reject_stale_lifecycle_event(&stale, rid, active).expect("stale");
        assert!(err.to_string().contains("stale attempt"), "{err}");

        let missing = serde_json::json!({"type":"token","id":"r","text":"x"});
        let err = reject_stale_lifecycle_event(&missing, rid, active).expect("missing");
        assert!(err.to_string().contains("missing attempt_id"), "{err}");

        let malformed = serde_json::json!({"type":"gen_start","id":"r","attempt_id":"2"});
        let err = reject_stale_lifecycle_event(&malformed, rid, active).expect("malformed");
        assert!(err.to_string().contains("malformed attempt_id"), "{err}");

        let wrong_id = serde_json::json!({"type":"token","id":"other","text":"x","attempt_id":2});
        let err = reject_stale_lifecycle_event(&wrong_id, rid, active).expect("wrong id");
        assert!(err.to_string().contains("stale id"), "{err}");

        let missing_id = serde_json::json!({"type":"token","text":"x","attempt_id":2});
        let err = reject_stale_lifecycle_event(&missing_id, rid, active).expect("missing id");
        assert!(err.to_string().contains("missing id"), "{err}");

        let empty_id = serde_json::json!({"type":"token","id":"","text":"x","attempt_id":2});
        let err = reject_stale_lifecycle_event(&empty_id, rid, active).expect("empty id");
        assert!(err.to_string().contains("empty id"), "{err}");

        for ty in [
            "gen_start",
            "token",
            "reasoning",
            "tool_calls",
            "committed",
            "commit_ready",
            "aborted",
            "done",
            "error",
        ] {
            let ok = serde_json::json!({"type": ty, "id": "r", "attempt_id": 2});
            assert!(
                reject_stale_lifecycle_event(&ok, rid, active).is_none(),
                "type={ty}"
            );
            let miss = serde_json::json!({"type": ty, "id": "r"});
            assert!(
                reject_stale_lifecycle_event(&miss, rid, active)
                    .unwrap()
                    .to_string()
                    .contains("missing attempt_id"),
                "type={ty}"
            );
            let bad_id = serde_json::json!({"type": ty, "id": "other", "attempt_id": 2});
            assert!(
                reject_stale_lifecycle_event(&bad_id, rid, active)
                    .unwrap()
                    .to_string()
                    .contains("stale id"),
                "type={ty}"
            );
        }

        // Without active attempt, no enforcement (non-generate traffic).
        assert!(reject_stale_lifecycle_event(&stale, rid, None).is_none());
    }

    #[test]
    fn require_attempt_id_rejects_missing_and_string() {
        assert_eq!(require_attempt_id(None).unwrap_err(), "missing attempt_id");
        assert_eq!(
            require_attempt_id(Some(&Value::String("3".into()))).unwrap_err(),
            "malformed attempt_id"
        );
        assert_eq!(require_attempt_id(Some(&Value::from(3u64))).unwrap(), 3);
    }

    #[test]
    fn route_lifecycle_idless_error_is_control() {
        let err = serde_json::json!({
            "type": "error",
            "message": "load failed",
            "class": error_class::INTERNAL,
            "retryable": false,
            "rolled_back": false,
            "attempt_id": 0,
        });
        assert_eq!(
            route_lifecycle_event("error", &err),
            LifecycleRoute::Control
        );

        let empty_id = serde_json::json!({
            "type": "error",
            "id": "",
            "message": "reset failed",
            "attempt_id": 1,
        });
        assert_eq!(
            route_lifecycle_event("error", &empty_id),
            LifecycleRoute::Control
        );

        let missing_attempt = serde_json::json!({
            "type": "error",
            "id": "req-x",
            "message": "no attempt",
        });
        assert_eq!(
            route_lifecycle_event("error", &missing_attempt),
            LifecycleRoute::Control
        );
    }

    #[test]
    fn route_lifecycle_keyed_error_is_pending_only() {
        let err = serde_json::json!({
            "type": "error",
            "id": "req-g",
            "message": "gen fail",
            "class": error_class::INTERNAL,
            "retryable": false,
            "rolled_back": false,
            "attempt_id": 7,
        });
        assert_eq!(
            route_lifecycle_event("error", &err),
            LifecycleRoute::Pending(("req-g".into(), 7))
        );
    }

    #[test]
    fn route_lifecycle_malformed_token_is_dropped() {
        let missing_id = serde_json::json!({"type":"token","text":"x","attempt_id":1});
        assert_eq!(
            route_lifecycle_event("token", &missing_id),
            LifecycleRoute::Drop
        );
        let empty_id = serde_json::json!({"type":"token","id":"","text":"x","attempt_id":1});
        assert_eq!(
            route_lifecycle_event("token", &empty_id),
            LifecycleRoute::Drop
        );
        let bad_attempt = serde_json::json!({"type":"token","id":"r","text":"x","attempt_id":"1"});
        assert_eq!(
            route_lifecycle_event("token", &bad_attempt),
            LifecycleRoute::Drop
        );
        let missing_attempt = serde_json::json!({"type":"token","id":"r","text":"x"});
        assert_eq!(
            route_lifecycle_event("token", &missing_attempt),
            LifecycleRoute::Drop
        );
        // Non-error lifecycle never escalates to control when unkeyed.
        for ty in [
            "gen_start",
            "token",
            "reasoning",
            "tool_calls",
            "committed",
            "commit_ready",
            "aborted",
            "done",
        ] {
            let v = serde_json::json!({"type": ty});
            assert_eq!(
                route_lifecycle_event(ty, &v),
                LifecycleRoute::Drop,
                "type={ty}"
            );
        }
    }

    #[test]
    fn generate_rejects_duplicate_live_key_preserves_sender() {
        let engine = dummy_engine();
        let (orig_tx, orig_rx) = mpsc::channel();
        let key = ("req-dup".to_owned(), 3u64);
        {
            let mut map = engine.inner.dispatch.pending.lock().unwrap();
            map.insert(key.clone(), orig_tx);
        }
        *engine.inner.active_attempt_id.lock().unwrap() = Some(3);

        let err = engine
            .generate(
                &serde_json::json!({"type":"generate","id":"req-dup","attempt_id":3}),
                |_| Ok(()),
            )
            .unwrap_err();
        assert!(
            matches!(&err, ClientError::Protocol(m) if m.contains("duplicate live generate")),
            "{err}"
        );

        // Original live registry entry and active-attempt bookkeeping stay put.
        assert_eq!(engine.active_attempt_id(), Some(3));
        {
            let map = engine.inner.dispatch.pending.lock().unwrap();
            assert!(map.contains_key(&key), "original pending entry must remain");
            assert_eq!(map.len(), 1);
        }
        // Original sender still delivers; the rejected call did not replace it.
        let marker =
            serde_json::json!({"type":"token","id":"req-dup","attempt_id":3,"text":"live"});
        {
            let map = engine.inner.dispatch.pending.lock().unwrap();
            let sender = map.get(&key).expect("original sender");
            sender.send(marker.clone()).expect("original sender live");
        }
        let got = orig_rx.try_recv().expect("original channel receives");
        assert_eq!(got, marker);
    }

    #[test]
    fn eligibility_comes_from_daemon_field_not_client_allowlist() {
        // No arch allowlist API on the client.
        let good = serde_json::json!({"retry_reset_eligible": true});
        assert_eq!(validate_retry_reset_eligible_field(&good).unwrap(), true);
        let bad = serde_json::json!({"retry_reset_eligible": false});
        assert_eq!(validate_retry_reset_eligible_field(&bad).unwrap(), false);
        let missing = serde_json::json!({"arch":"qwen35"});
        assert!(validate_retry_reset_eligible_field(&missing).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn reset_ack_full_validation_via_fake_daemon() {
        use std::os::unix::fs::PermissionsExt;
        let root = env::temp_dir().join(format!("hipfire-client-reset-ack-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let daemon = root.join("daemon");
        // First reset: full valid ack. Second: type-only legacy (must fail).
        // Third: valid shape but wrong attempt echo.
        fs::write(
            &daemon,
            r#"#!/bin/sh
n=0
while IFS= read -r line; do
 case "$line" in
  *'"reset"'*)
    n=$((n+1))
    if [ "$n" -eq 1 ]; then
      echo '{"type":"reset","rolled_back":true,"state_epoch":1,"seq_pos":0,"conversation_len":0,"attempt_id":5,"retry_reset_eligible":true}'
    elif [ "$n" -eq 2 ]; then
      echo '{"type":"reset","seq_pos":0}'
    else
      echo '{"type":"reset","rolled_back":true,"state_epoch":2,"seq_pos":0,"conversation_len":0,"attempt_id":99,"retry_reset_eligible":false}'
    fi
    ;;
  *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
 esac
done
"#,
        )
        .unwrap();
        fs::set_permissions(&daemon, fs::Permissions::from_mode(0o755)).unwrap();
        let mut engine = spawn_fake_engine(&daemon);
        engine.reset(5).unwrap();
        assert_eq!(engine.last_state_epoch(), Some(1));
        assert_eq!(engine.last_retry_reset_eligible(), Some(true));
        let err = engine.reset(6).unwrap_err();
        assert!(matches!(err, ClientError::Protocol(_)), "{err}");
        let err = engine.reset(7).unwrap_err();
        assert!(err.to_string().contains("attempt_id mismatch"), "{err}");
        engine.unload().unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_requires_retry_reset_eligible_and_clears_stale() {
        // Unit-level: after a prior eligible=true, a loaded response missing the
        // field must not keep the stale bit; validate_retry_reset_eligible_field
        // is the fail-closed gate used by Engine::load.
        let mut engine = dummy_engine();
        *engine.inner.last_retry_reset_eligible.lock().unwrap() = Some(true);

        // Simulate pre-load clear (same as Engine::load entry).
        *engine.inner.last_retry_reset_eligible.lock().unwrap() = None;
        assert_eq!(engine.last_retry_reset_eligible(), None);

        let missing = serde_json::json!({
            "type": "loaded",
            "arch": "qwen35",
            "dim": 1,
            "layers": 1,
            "vocab": 1,
            "vl": false,
            "cache_capable": false
        });
        let err = validate_retry_reset_eligible_field(&missing).unwrap_err();
        assert!(err.to_string().contains("retry_reset_eligible"), "{err}");
        // Eligibility stays cleared on validation failure (fail closed).
        assert_eq!(engine.last_retry_reset_eligible(), None);

        let mistyped = serde_json::json!({
            "type": "loaded",
            "retry_reset_eligible": "yes"
        });
        assert!(validate_retry_reset_eligible_field(&mistyped).is_err());

        let good = serde_json::json!({
            "type": "loaded",
            "retry_reset_eligible": false
        });
        let eligible = validate_retry_reset_eligible_field(&good).unwrap();
        *engine.inner.last_retry_reset_eligible.lock().unwrap() = Some(eligible);
        assert_eq!(engine.last_retry_reset_eligible(), Some(false));
    }

    #[cfg(unix)]
    #[test]
    fn load_eligible_then_malformed_clears_and_fails() {
        use std::os::unix::fs::PermissionsExt;
        let root = env::temp_dir().join(format!("hipfire-client-load-elig-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let daemon = root.join("daemon");
        // First load: valid eligible=true. Second: omits field (must fail closed
        // and clear prior eligibility).
        fs::write(
            &daemon,
            r#"#!/bin/sh
n=0
while IFS= read -r line; do
 case "$line" in
  *'"load"'*)
    n=$((n+1))
    if [ "$n" -eq 1 ]; then
      echo '{"type":"loaded","arch":"qwen35","dim":1,"layers":1,"vocab":1,"vl":false,"cache_capable":true,"retry_reset_eligible":true}'
    else
      echo '{"type":"loaded","arch":"qwen35","dim":1,"layers":1,"vocab":1,"vl":false,"cache_capable":true}'
    fi
    ;;
  *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
 esac
done
"#,
        )
        .unwrap();
        fs::set_permissions(&daemon, fs::Permissions::from_mode(0o755)).unwrap();
        let mut engine = spawn_fake_engine(&daemon);
        let loaded = engine
            .load(Path::new("/tmp/m.hfq"), serde_json::json!({}))
            .unwrap();
        assert_eq!(
            loaded.get("retry_reset_eligible").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(engine.last_retry_reset_eligible(), Some(true));

        let err = engine
            .load(Path::new("/tmp/m2.hfq"), serde_json::json!({}))
            .unwrap_err();
        assert!(err.to_string().contains("retry_reset_eligible"), "{err}");
        // Prior eligibility must not stick after a malformed loaded response.
        assert_eq!(engine.last_retry_reset_eligible(), None);
        engine.unload().unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn typed_error_attempt_id_echoes_active_nonzero() {
        // Once a numeric generate attempt is known, validation/generation
        // errors must carry that exact ID (not hard-coded 0).
        for attempt in [1u64, 7, 42] {
            for class in [
                error_class::VALIDATION,
                error_class::MALFORMED,
                error_class::TRANSIENT,
                error_class::INTERNAL,
            ] {
                let ev = serde_json::json!({
                    "type": "error",
                    "id": "r1",
                    "message": "invalid tools field: bad",
                    "class": class,
                    "retryable": false,
                    "rolled_back": false,
                    "attempt_id": attempt,
                });
                let err = daemon_error_from_value(&ev);
                let typed = err.typed_daemon().expect("typed");
                assert_eq!(typed.attempt_id, attempt, "class={class} attempt={attempt}");
            }
        }

        // Protocol rejection when attempt_id itself is missing may use 0.
        let uncorrelated = serde_json::json!({
            "type": "error",
            "message": "generate missing attempt_id",
            "class": error_class::VALIDATION,
            "retryable": false,
            "rolled_back": false,
            "attempt_id": 0,
        });
        let err = daemon_error_from_value(&uncorrelated);
        let typed = err.typed_daemon().expect("typed");
        assert_eq!(typed.attempt_id, 0);
    }

    #[cfg(unix)]
    fn write_fake_daemon(root: &std::path::Path, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        // Drop any prior root so we never truncate an inode still executing
        // from a PID-reused parallel test (ETXTBSY).
        let _ = fs::remove_dir_all(root);
        fs::create_dir_all(root).unwrap();
        let daemon = root.join("daemon");
        fs::write(&daemon, script).unwrap();
        fs::set_permissions(&daemon, fs::Permissions::from_mode(0o755)).unwrap();
        daemon
    }

    /// Test-only spawn with bounded ETXTBSY retry (parallel PID-reused scripts).
    #[cfg(unix)]
    fn spawn_fake_engine(daemon: &std::path::Path) -> Engine {
        const ETXTBSY: i32 = 26;
        const MAX_ATTEMPTS: usize = 8;
        let mut last = None;
        for attempt in 0..MAX_ATTEMPTS {
            match Engine::spawn(daemon, &BTreeMap::new()) {
                Ok(engine) => return engine,
                Err(ClientError::Spawn { source, path })
                    if source.raw_os_error() == Some(ETXTBSY) =>
                {
                    last = Some(ClientError::Spawn { path, source });
                    std::thread::sleep(std::time::Duration::from_millis(
                        5u64.saturating_mul(1 + attempt as u64),
                    ));
                }
                Err(err) => panic!("spawn_fake_engine non-retryable: {err}"),
            }
        }
        panic!(
            "spawn_fake_engine exhausted ETXTBSY retries: {}",
            last.map(|e| e.to_string()).unwrap_or_default()
        );
    }

    /// Same bounded ETXTBSY retry as `spawn_fake_engine`, for the configured path.
    #[cfg(unix)]
    fn spawn_fake_engine_configured(
        daemon: &std::path::Path,
        config: &hipfire_config::ProcessConfig,
    ) -> Engine {
        const ETXTBSY: i32 = 26;
        const MAX_ATTEMPTS: usize = 8;
        let mut last = None;
        for attempt in 0..MAX_ATTEMPTS {
            match Engine::spawn_configured(daemon, &BTreeMap::new(), config) {
                Ok(engine) => return engine,
                Err(ClientError::Spawn { source, path })
                    if source.raw_os_error() == Some(ETXTBSY) =>
                {
                    last = Some(ClientError::Spawn { path, source });
                    std::thread::sleep(std::time::Duration::from_millis(
                        5u64.saturating_mul(1 + attempt as u64),
                    ));
                }
                Err(err) => panic!("spawn_fake_engine_configured non-retryable: {err}"),
            }
        }
        panic!(
            "spawn_fake_engine_configured exhausted ETXTBSY retries: {}",
            last.map(|e| e.to_string()).unwrap_or_default()
        );
    }

    #[cfg(unix)]
    #[test]
    fn generate_commit_ready_sends_matching_commit() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-commit-ack-{}-{}",
            std::process::id(),
            "a"
        ));
        let log = root.join("in.log");
        let daemon = write_fake_daemon(
            &root,
            &format!(
                r#"#!/bin/sh
LOG="{log}"
: > "$LOG"
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$LOG"
  case "$line" in
    *'"generate"'*)
      echo '{{"type":"gen_start","id":"req-1","attempt_id":7}}'
      echo '{{"type":"commit_ready","id":"req-1","attempt_id":7,"finish_reason":"stop"}}'
      ;;
    *'"commit"'*)
      echo '{{"type":"done","id":"req-1","attempt_id":7,"finish_reason":"stop"}}'
      ;;
    *'"unload"'*) echo '{{"type":"unloaded"}}'; exit 0 ;;
  esac
done
"#,
                log = log.display()
            ),
        );
        let mut engine = spawn_fake_engine(&daemon);
        let mut callbacks = 0usize;
        let mut saw_ready = false;
        let mut saw_done_in_cb = false;
        let done = engine
            .generate(
                &serde_json::json!({
                    "type": "generate",
                    "id": "req-1",
                    "attempt_id": 7,
                    "prompt": "hi",
                }),
                |ev| {
                    callbacks += 1;
                    match ev.get("type").and_then(Value::as_str) {
                        Some("commit_ready") => saw_ready = true,
                        Some("done") => saw_done_in_cb = true,
                        _ => {}
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert!(saw_ready);
        assert!(!saw_done_in_cb, "final done must not be callback-visible");
        assert_eq!(callbacks, 2);
        assert_eq!(
            done.get("finish_reason").and_then(Value::as_str),
            Some("stop")
        );
        assert_eq!(engine.active_attempt_id(), None);

        let logged = fs::read_to_string(&log).unwrap();
        assert!(
            logged.contains(r#""type":"commit""#)
                && logged.contains(r#""id":"req-1""#)
                && logged.contains(r#""attempt_id":7"#),
            "commit control missing: {logged}"
        );
        assert!(
            !logged.contains(r#""type":"abort""#),
            "successful commit must not send abort: {logged}"
        );
        engine.unload().unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_cancelled_callback_sends_abort_and_drains() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-cancel-drain-{}-{}",
            std::process::id(),
            "b"
        ));
        let log = root.join("in.log");
        let daemon = write_fake_daemon(
            &root,
            &format!(
                r#"#!/bin/sh
LOG="{log}"
: > "$LOG"
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$LOG"
  case "$line" in
    *'"generate"'*)
      echo '{{"type":"token","id":"req-c","text":"x","attempt_id":3}}'
      echo '{{"type":"commit_ready","id":"req-c","attempt_id":3}}'
      ;;
    *'"abort"'*)
      echo '{{"type":"aborted","id":"req-c","reason":"client_cancelled","attempt_id":3}}'
      echo '{{"type":"done","id":"req-c","finish_reason":"aborted","attempt_id":3,"completion_tokens":1}}'
      ;;
    *'"commit"'*)
      echo '{{"type":"error","message":"unexpected commit after cancel","class":"internal","retryable":false,"rolled_back":false,"attempt_id":3}}'
      ;;
    *'"unload"'*) echo '{{"type":"unloaded"}}'; exit 0 ;;
  esac
done
"#,
                log = log.display()
            ),
        );
        let mut engine = spawn_fake_engine(&daemon);
        let mut callbacks = 0usize;
        let err = engine
            .generate(
                &serde_json::json!({
                    "type": "generate",
                    "id": "req-c",
                    "attempt_id": 3,
                }),
                |ev| {
                    callbacks += 1;
                    if ev.get("type").and_then(Value::as_str) == Some("commit_ready") {
                        return Err(ClientError::Cancelled);
                    }
                    Ok(())
                },
            )
            .unwrap_err();
        assert!(matches!(err, ClientError::Cancelled), "{err}");
        // token + commit_ready only — drain must not re-invoke callback.
        assert_eq!(callbacks, 2);
        assert_eq!(engine.active_attempt_id(), None);

        let logged = fs::read_to_string(&log).unwrap();
        assert!(
            logged.lines().any(|l| {
                l.contains(r#""type":"abort""#)
                    && l.contains(r#""id":"req-c""#)
                    && l.contains(r#""attempt_id":3"#)
            }),
            "abort control missing: {logged}"
        );
        assert!(
            !logged.contains(r#""type":"commit""#),
            "must not commit after cancel: {logged}"
        );
        engine.unload().unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_terminal_error_must_not_abort_or_hang() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-terminal-error-{}-{}",
            std::process::id(),
            "te"
        ));
        let log = root.join("in.log");
        let daemon = write_fake_daemon(
            &root,
            &format!(
                r#"#!/bin/sh
LOG="{log}"
: > "$LOG"
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$LOG"
  case "$line" in
    *'"generate"'*)
      echo '{{"type":"error","id":"req-te","message":"request exceeds loaded KV budget","class":"context_length","retryable":false,"rolled_back":false,"attempt_id":7}}'
      ;;
    *'"unload"'*) echo '{{"type":"unloaded"}}'; exit 0 ;;
  esac
done
"#,
                log = log.display()
            ),
        );

        let (tx, rx) = std::sync::mpsc::channel();
        let worker_daemon = daemon.clone();
        std::thread::spawn(move || {
            let mut engine = spawn_fake_engine(&worker_daemon);
            let outcome = engine.generate(
                &serde_json::json!({
                    "type": "generate",
                    "id": "req-te",
                    "attempt_id": 7,
                }),
                |ev| match ev.get("type").and_then(Value::as_str) {
                    Some("error") => Err(ClientError::Cancelled),
                    _ => Ok(()),
                },
            );
            let _ = tx.send(format!("{outcome:?}"));
            let _ = engine.unload();
        });

        let outcome = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("generate never returned: abort_and_drain waits for a terminal pair the daemon will not send");

        assert!(
            outcome.contains("Cancelled"),
            "callback error must propagate unchanged: {outcome}"
        );
        let logged = fs::read_to_string(&log).unwrap_or_default();
        assert!(
            !logged.contains(r#""type":"abort""#),
            "must not abort a transaction the daemon already terminated: {logged}"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_non_cancelled_callback_error_also_aborts() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-cb-err-abort-{}-{}",
            std::process::id(),
            "c"
        ));
        let log = root.join("in.log");
        let daemon = write_fake_daemon(
            &root,
            &format!(
                r#"#!/bin/sh
LOG="{log}"
: > "$LOG"
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$LOG"
  case "$line" in
    *'"generate"'*)
      echo '{{"type":"commit_ready","id":"req-e","attempt_id":9}}'
      ;;
    *'"abort"'*)
      echo '{{"type":"aborted","id":"req-e","reason":"client_cancelled","attempt_id":9}}'
      echo '{{"type":"done","id":"req-e","finish_reason":"aborted","attempt_id":9}}'
      ;;
    *'"unload"'*) echo '{{"type":"unloaded"}}'; exit 0 ;;
  esac
done
"#,
                log = log.display()
            ),
        );
        let mut engine = spawn_fake_engine(&daemon);
        let err = engine
            .generate(
                &serde_json::json!({
                    "type": "generate",
                    "id": "req-e",
                    "attempt_id": 9,
                }),
                |_| Err(ClientError::Http("sse write failed".into())),
            )
            .unwrap_err();
        assert!(
            matches!(&err, ClientError::Http(m) if m.contains("sse write failed")),
            "{err}"
        );
        assert_eq!(engine.active_attempt_id(), None);
        let logged = fs::read_to_string(&log).unwrap();
        assert!(
            logged.contains(r#""type":"abort""#) && logged.contains(r#""attempt_id":9"#),
            "{logged}"
        );
        engine.unload().unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_drain_rejects_stale_attempt_event() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-drain-stale-{}-{}",
            std::process::id(),
            "d"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"generate"'*)
      echo '{"type":"commit_ready","id":"req-s","attempt_id":4}'
      ;;
    *'"abort"'*)
      echo '{"type":"aborted","id":"req-s","reason":"client_cancelled","attempt_id":4}'
      echo '{"type":"done","id":"req-s","finish_reason":"aborted","attempt_id":4}'
      ;;
    *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
  esac
done
"#,
        );
        let mut engine = spawn_fake_engine(&daemon);
        let err = engine
            .generate(
                &serde_json::json!({
                    "type": "generate",
                    "id": "req-s",
                    "attempt_id": 4,
                }),
                |_| Err(ClientError::Cancelled),
            )
            .unwrap_err();
        assert!(
            matches!(&err, ClientError::Cancelled),
            "client cancellation must survive after stale drain events: {err}"
        );
        assert_eq!(engine.active_attempt_id(), None);
        // Protocol poison left a follow-on done queued — Drop kills the child.
        drop(engine);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_drain_rejects_missing_attempt_on_aborted() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-drain-missing-{}-{}",
            std::process::id(),
            "e"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"generate"'*)
      echo '{"type":"commit_ready","id":"req-m","attempt_id":5}'
      ;;
    *'"abort"'*)
      echo '{"type":"aborted","id":"req-m","reason":"client_cancelled"}'
      echo '{"type":"done","id":"req-m","finish_reason":"aborted","attempt_id":5}'
      ;;
    *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
  esac
done
"#,
        );
        let mut engine = spawn_fake_engine(&daemon);
        let err = engine
            .generate(
                &serde_json::json!({
                    "type": "generate",
                    "id": "req-m",
                    "attempt_id": 5,
                }),
                |_| Err(ClientError::Cancelled),
            )
            .unwrap_err();
        assert!(
            matches!(&err, ClientError::Cancelled),
            "malformed aborted event is dropped; valid done completes cancellation: {err}"
        );
        assert_eq!(engine.active_attempt_id(), None);
        // Protocol poison left a follow-on done queued — Drop kills the child.
        drop(engine);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_drain_error_only_rollback_propagates() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-drain-err-{}-{}",
            std::process::id(),
            "f"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"generate"'*)
      echo '{"type":"commit_ready","id":"req-r","attempt_id":2}'
      ;;
    *'"abort"'*)
      echo '{"type":"error","id":"req-r","message":"rollback failed","class":"internal","retryable":false,"rolled_back":false,"attempt_id":2}'
      ;;
    *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
  esac
done
"#,
        );
        let mut engine = spawn_fake_engine(&daemon);
        let err = engine
            .generate(
                &serde_json::json!({
                    "type": "generate",
                    "id": "req-r",
                    "attempt_id": 2,
                }),
                |_| Err(ClientError::Cancelled),
            )
            .unwrap_err();
        let typed = err.typed_daemon().expect("typed daemon error");
        assert_eq!(typed.message, "rollback failed");
        assert!(!typed.rolled_back);
        assert_eq!(typed.attempt_id, 2);
        assert_eq!(engine.active_attempt_id(), None);
        engine.unload().unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_drain_eof_propagates() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-drain-eof-{}-{}",
            std::process::id(),
            "g"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"generate"'*)
      echo '{"type":"commit_ready","id":"req-eof","attempt_id":1}'
      ;;
    *'"abort"'*)
      exit 0
      ;;
  esac
done
"#,
        );
        let mut engine = spawn_fake_engine(&daemon);
        let err = engine
            .generate(
                &serde_json::json!({
                    "type": "generate",
                    "id": "req-eof",
                    "attempt_id": 1,
                }),
                |_| Err(ClientError::Cancelled),
            )
            .unwrap_err();
        assert!(matches!(err, ClientError::Closed { .. }), "{err}");
        assert_eq!(engine.active_attempt_id(), None);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_drain_rejects_normal_done_as_cancellation() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-drain-normal-done-{}-{}",
            std::process::id(),
            "h"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"generate"'*)
      echo '{"type":"commit_ready","id":"req-n","attempt_id":6}'
      ;;
    *'"abort"'*)
      echo '{"type":"done","id":"req-n","finish_reason":"stop","attempt_id":6}'
      ;;
    *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
  esac
done
"#,
        );
        let mut engine = spawn_fake_engine(&daemon);
        let err = engine
            .generate(
                &serde_json::json!({
                    "type": "generate",
                    "id": "req-n",
                    "attempt_id": 6,
                }),
                |_| Err(ClientError::Cancelled),
            )
            .unwrap_err();
        assert!(
            matches!(
                &err,
                ClientError::Protocol(m) if m.contains("non-aborted done")
            ),
            "{err}"
        );
        assert_eq!(engine.active_attempt_id(), None);
        engine.unload().unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_requires_request_id_and_keeps_active_until_terminal() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-active-until-{}-{}",
            std::process::id(),
            "i"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"generate"'*)
      echo '{"type":"token","id":"req-a","text":"t","attempt_id":8}'
      echo '{"type":"commit_ready","id":"req-a","attempt_id":8}'
      ;;
    *'"abort"'*)
      # Delay terminal so active_attempt is still set when we observe mid-flight.
      sleep 0.05
      echo '{"type":"aborted","id":"req-a","reason":"client_cancelled","attempt_id":8}'
      echo '{"type":"done","id":"req-a","finish_reason":"aborted","attempt_id":8}'
      ;;
    *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
  esac
done
"#,
        );
        let mut engine = spawn_fake_engine(&daemon);

        let err = engine
            .generate(
                &serde_json::json!({
                    "type": "generate",
                    "attempt_id": 1,
                }),
                |_| Ok(()),
            )
            .unwrap_err();
        assert!(
            matches!(&err, ClientError::Protocol(m) if m.contains("missing id")),
            "{err}"
        );
        assert_eq!(engine.active_attempt_id(), None);

        // After cancel, active clears only once drain terminal is accepted —
        // proven by successful return of Cancelled with active cleared.
        let mut saw_active_during_callback = false;
        let err = engine
            .generate(
                &serde_json::json!({
                    "type": "generate",
                    "id": "req-a",
                    "attempt_id": 8,
                }),
                |ev| {
                    // active_attempt_id is internal; correlation is enforced on
                    // the wire. Observe that we still receive events (active).
                    if ev.get("attempt_id").and_then(Value::as_u64) == Some(8) {
                        saw_active_during_callback = true;
                    }
                    if ev.get("type").and_then(Value::as_str) == Some("commit_ready") {
                        return Err(ClientError::Cancelled);
                    }
                    Ok(())
                },
            )
            .unwrap_err();
        assert!(matches!(err, ClientError::Cancelled), "{err}");
        assert!(saw_active_during_callback);
        // Cleared only after terminal drain completed.
        assert_eq!(engine.active_attempt_id(), None);
        engine.unload().unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn committed_done_payload_must_match_staged_ready() {
        let ready = serde_json::json!({
            "type": "commit_ready",
            "id": "r1",
            "attempt_id": 3,
            "finish_reason": "stop",
        });
        let matching = serde_json::json!({
            "type": "done",
            "id": "r1",
            "attempt_id": 3,
            "finish_reason": "stop",
        });
        assert!(committed_done_payload_mismatch(&ready, &matching).is_none());
        let diverged = serde_json::json!({
            "type": "done",
            "id": "r1",
            "attempt_id": 3,
            "finish_reason": "length",
        });
        let err = committed_done_payload_mismatch(&ready, &diverged).expect("mismatch");
        assert!(
            matches!(&err, ClientError::Protocol(m) if m.contains("diverges")),
            "{err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn generate_rejects_wrong_id_lifecycle_events() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-wrong-id-{}-{}",
            std::process::id(),
            "w"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"generate"'*)
      echo '{"type":"token","id":"other","text":"x","attempt_id":1}'
      echo '{"type":"commit_ready","id":"req-w","attempt_id":1,"finish_reason":"stop"}'
      ;;
    *'"commit"'*)
      echo '{"type":"done","id":"req-w","attempt_id":1,"finish_reason":"stop"}'
      ;;
    *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
  esac
done
"#,
        );
        let mut engine = spawn_fake_engine(&daemon);
        let done = engine
            .generate(
                &serde_json::json!({"type":"generate","id":"req-w","attempt_id":1}),
                |_| Ok(()),
            )
            .expect("unknown wrong-id token is dropped and normal ready/done completes");
        assert_eq!(
            done.get("finish_reason").and_then(|v| v.as_str()),
            Some("stop")
        );
        assert_eq!(engine.active_attempt_id(), None);
        drop(engine);
        let _ = fs::remove_dir_all(&root);

        let root = env::temp_dir().join(format!(
            "hipfire-client-wrong-id-ready-{}-{}",
            std::process::id(),
            "w2"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"generate"'*)
      echo '{"type":"commit_ready","id":"other","attempt_id":2,"finish_reason":"stop"}'
      echo '{"type":"commit_ready","id":"req-w2","attempt_id":2,"finish_reason":"stop"}'
      ;;
    *'"commit"'*)
      echo '{"type":"done","id":"req-w2","attempt_id":2,"finish_reason":"stop"}'
      ;;
    *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
  esac
done
"#,
        );
        let mut engine = spawn_fake_engine(&daemon);
        let mut callbacks = 0usize;
        let done = engine
            .generate(
                &serde_json::json!({"type":"generate","id":"req-w2","attempt_id":2}),
                |_| {
                    callbacks += 1;
                    Ok(())
                },
            )
            .expect("unknown wrong-id commit_ready is dropped and own ready/done completes");
        assert_eq!(
            done.get("finish_reason").and_then(|v| v.as_str()),
            Some("stop")
        );
        assert_eq!(callbacks, 1);
        drop(engine);
        let _ = fs::remove_dir_all(&root);

        let root = env::temp_dir().join(format!(
            "hipfire-client-wrong-id-abort-{}-{}",
            std::process::id(),
            "wa"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"generate"'*)
      echo '{"type":"commit_ready","id":"req-wa","attempt_id":4,"finish_reason":"stop"}'
      ;;
    *'"abort"'*)
      echo '{"type":"aborted","id":"other","reason":"client_cancelled","attempt_id":4}'
      echo '{"type":"aborted","id":"req-wa","reason":"client_cancelled","attempt_id":4}'
      echo '{"type":"done","id":"req-wa","finish_reason":"aborted","attempt_id":4}'
      ;;
    *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
  esac
done
"#,
        );
        let mut engine = spawn_fake_engine(&daemon);
        let err = engine
            .generate(
                &serde_json::json!({"type":"generate","id":"req-wa","attempt_id":4}),
                |_| Err(ClientError::Cancelled),
            )
            .unwrap_err();
        assert!(
            matches!(&err, ClientError::Cancelled),
            "unknown aborted response is quarantined; client cancellation survives: {err}"
        );
        drop(engine);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_post_commit_no_abort_on_mismatch_or_unexpected() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-post-commit-{}-{}",
            std::process::id(),
            "pc"
        ));
        let log = root.join("in.log");
        let daemon = write_fake_daemon(
            &root,
            &format!(
                r#"#!/bin/sh
LOG="{log}"
: > "$LOG"
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$LOG"
  case "$line" in
    *'"generate"'*)
      echo '{{"type":"commit_ready","id":"req-pc","attempt_id":11,"finish_reason":"stop"}}'
      ;;
    *'"commit"'*)
      echo '{{"type":"done","id":"req-pc","attempt_id":11,"finish_reason":"length"}}'
      ;;
    *'"abort"'*) echo 'ABORT_SENT' >> "$LOG" ;;
    *'"unload"'*) echo '{{"type":"unloaded"}}'; exit 0 ;;
  esac
done
"#,
                log = log.display()
            ),
        );
        let mut engine = spawn_fake_engine(&daemon);
        let mut callbacks = 0usize;
        let err = engine
            .generate(
                &serde_json::json!({"type":"generate","id":"req-pc","attempt_id":11}),
                |_| {
                    callbacks += 1;
                    Ok(())
                },
            )
            .unwrap_err();
        assert!(
            matches!(&err, ClientError::Protocol(m) if m.contains("diverges")),
            "{err}"
        );
        assert_eq!(callbacks, 1);
        let logged = fs::read_to_string(&log).unwrap();
        assert!(logged.contains(r#""type":"commit""#), "{logged}");
        assert!(
            !logged.contains("ABORT_SENT") && !logged.contains(r#""type":"abort""#),
            "{logged}"
        );
        drop(engine);
        let _ = fs::remove_dir_all(&root);

        let root = env::temp_dir().join(format!(
            "hipfire-client-post-commit-err-{}-{}",
            std::process::id(),
            "pe"
        ));
        let log = root.join("in.log");
        let daemon = write_fake_daemon(
            &root,
            &format!(
                r#"#!/bin/sh
LOG="{log}"
: > "$LOG"
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$LOG"
  case "$line" in
    *'"generate"'*)
      echo '{{"type":"commit_ready","id":"req-pe","attempt_id":12,"finish_reason":"stop"}}'
      ;;
    *'"commit"'*)
      echo '{{"type":"error","id":"req-pe","message":"post-commit fail","class":"internal","retryable":false,"rolled_back":false,"attempt_id":12}}'
      ;;
    *'"abort"'*) echo 'ABORT_SENT' >> "$LOG" ;;
    *'"unload"'*) echo '{{"type":"unloaded"}}'; exit 0 ;;
  esac
done
"#,
                log = log.display()
            ),
        );
        let mut engine = spawn_fake_engine(&daemon);
        let err = engine
            .generate(
                &serde_json::json!({"type":"generate","id":"req-pe","attempt_id":12}),
                |_| Ok(()),
            )
            .unwrap_err();
        assert!(err.typed_daemon().is_some(), "{err}");
        let logged = fs::read_to_string(&log).unwrap();
        assert!(logged.contains(r#""type":"commit""#), "{logged}");
        assert!(
            !logged.contains("ABORT_SENT") && !logged.contains(r#""type":"abort""#),
            "{logged}"
        );
        drop(engine);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_concurrent_routing_isolated() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-concurrent-{}-{}",
            std::process::id(),
            "conc"
        ));
        let log = root.join("in.log");
        let daemon = write_fake_daemon(
            &root,
            &format!(
                r#"#!/bin/sh
LOG="{log}"
: > "$LOG"
n=0
line1=""
line2=""
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$LOG"
  n=$((n+1))
  if [ "$n" -eq 1 ]; then line1="$line"; continue; fi
  if [ "$n" -eq 2 ]; then
    line2="$line"
    echo '{{"type":"gen_start","id":"req-a","attempt_id":1}}'
    echo '{{"type":"gen_start","id":"req-b","attempt_id":2}}'
    echo '{{"type":"token","id":"req-a","attempt_id":1,"text":"a1"}}'
    echo '{{"type":"token","id":"req-b","attempt_id":2,"text":"b1"}}'
    echo '{{"type":"commit_ready","id":"req-a","attempt_id":1,"finish_reason":"stop"}}'
    echo '{{"type":"commit_ready","id":"req-b","attempt_id":2,"finish_reason":"stop"}}'
    continue
  fi
  case "$line" in
    *'"type":"commit"'*)
      if echo "$line" | grep -q "req-a"; then
        echo '{{"type":"done","id":"req-a","attempt_id":1,"finish_reason":"stop"}}'
      else
        echo '{{"type":"done","id":"req-b","attempt_id":2,"finish_reason":"stop"}}'
      fi
      ;;
    *'"unload"'*) echo '{{"type":"unloaded"}}'; exit 0 ;;
  esac
done
"#,
                log = log.display()
            ),
        );
        let engine = std::sync::Arc::new(spawn_fake_engine(&daemon));
        let engine_a = std::sync::Arc::clone(&engine);
        let engine_b = std::sync::Arc::clone(&engine);
        let handle_a = std::thread::spawn(move || {
            let mut tokens = Vec::new();
            let done = engine_a
                .generate(
                    &serde_json::json!({"type":"generate","id":"req-a","attempt_id":1}),
                    |ev| {
                        if let Some(text) = ev.get("text").and_then(|v| v.as_str()) {
                            tokens.push(text.to_owned());
                        }
                        Ok(())
                    },
                )
                .expect("req-a generate ok");
            (tokens, done)
        });
        let handle_b = std::thread::spawn(move || {
            let mut tokens = Vec::new();
            let done = engine_b
                .generate(
                    &serde_json::json!({"type":"generate","id":"req-b","attempt_id":2}),
                    |ev| {
                        if let Some(text) = ev.get("text").and_then(|v| v.as_str()) {
                            tokens.push(text.to_owned());
                        }
                        Ok(())
                    },
                )
                .expect("req-b generate ok");
            (tokens, done)
        });
        let (tokens_a, done_a) = handle_a.join().expect("thread a");
        let (tokens_b, done_b) = handle_b.join().expect("thread b");
        assert_eq!(
            tokens_a,
            vec!["a1"],
            "req-a should receive only its own token"
        );
        assert_eq!(
            tokens_b,
            vec!["b1"],
            "req-b should receive only its own token"
        );
        assert_eq!(done_a.get("id").and_then(|v| v.as_str()), Some("req-a"));
        assert_eq!(done_b.get("id").and_then(|v| v.as_str()), Some("req-b"));
        // Ensure no cross-routing: each done id matches its request.
        drop(engine);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_stale_unknown_is_quarantined_not_routed() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-stale-{}-{}",
            std::process::id(),
            "stale"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"generate"'*)
      # Unknown keyed lifecycle is quarantined, not broadcast into this request.
      echo '{"type":"token","id":"other","attempt_id":1,"text":"bad"}'
      echo '{"type":"commit_ready","id":"req-s","attempt_id":1,"finish_reason":"stop"}'
      ;;
    *'"commit"'*) echo '{"type":"done","id":"req-s","attempt_id":1,"finish_reason":"stop"}' ;;
    *'"abort"'*)
      echo '{"type":"aborted","id":"req-s","reason":"client_cancelled","attempt_id":1}'
      echo '{"type":"done","id":"req-s","finish_reason":"aborted","attempt_id":1}'
      ;;
    *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
  esac
done
"#,
        );
        let engine = spawn_fake_engine(&daemon);
        let done = engine
            .generate(
                &serde_json::json!({"type":"generate","id":"req-s","attempt_id":1}),
                |_| Ok(()),
            )
            .expect("unknown stale lane must not poison this request");
        assert_eq!(
            done.get("finish_reason").and_then(|v| v.as_str()),
            Some("stop")
        );
        drop(engine);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn load_idless_error_reaches_control_plane() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-idless-err-{}-{}",
            std::process::id(),
            "idless"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"load"'*)
      # Control-plane error intentionally omits generation id.
      echo '{"type":"error","message":"bad model","class":"validation","retryable":false,"rolled_back":false,"attempt_id":0}'
      ;;
    *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
  esac
done
"#,
        );
        let engine = spawn_fake_engine(&daemon);
        let err = engine
            .load(std::path::Path::new("/tmp/model"), serde_json::json!({}))
            .unwrap_err();
        let typed = err.typed_daemon().expect("typed control error");
        assert_eq!(typed.message, "bad model");
        assert_eq!(typed.class, error_class::VALIDATION);
        drop(engine);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_keyed_error_isolated_from_control() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-keyed-err-{}-{}",
            std::process::id(),
            "keyed"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"generate"'*)
      echo '{"type":"error","id":"req-k","message":"gen boom","class":"internal","retryable":false,"rolled_back":false,"attempt_id":5}'
      ;;
    *'"ping"'*) echo '{"type":"pong"}' ;;
    *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
  esac
done
"#,
        );
        let engine = spawn_fake_engine(&daemon);
        let err = engine
            .generate(
                &serde_json::json!({"type":"generate","id":"req-k","attempt_id":5}),
                |_| Ok(()),
            )
            .unwrap_err();
        let typed = err.typed_daemon().expect("generation error");
        assert_eq!(typed.message, "gen boom");
        assert_eq!(typed.id.as_deref(), Some("req-k"));
        assert_eq!(typed.attempt_id, 5);
        // Control plane remains healthy after a keyed generation error.
        engine.ping().expect("control not poisoned by keyed error");
        drop(engine);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn generate_malformed_token_not_broadcast_to_live_request() {
        let root = env::temp_dir().join(format!(
            "hipfire-client-malformed-token-{}-{}",
            std::process::id(),
            "mft"
        ));
        let daemon = write_fake_daemon(
            &root,
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"generate"'*)
      # Unkeyed/malformed lifecycle must be dropped, not fanned out.
      echo '{"type":"token","text":"poison","attempt_id":1}'
      echo '{"type":"token","id":"","text":"poison2","attempt_id":1}'
      echo '{"type":"token","id":"req-m","text":"poison3","attempt_id":"1"}'
      echo '{"type":"commit_ready","id":"req-m","attempt_id":1,"finish_reason":"stop"}'
      ;;
    *'"commit"'*) echo '{"type":"done","id":"req-m","attempt_id":1,"finish_reason":"stop"}' ;;
    *'"unload"'*) echo '{"type":"unloaded"}'; exit 0 ;;
  esac
done
"#,
        );
        let engine = spawn_fake_engine(&daemon);
        let mut tokens = Vec::new();
        let done = engine
            .generate(
                &serde_json::json!({"type":"generate","id":"req-m","attempt_id":1}),
                |ev| {
                    if let Some(text) = ev.get("text").and_then(|v| v.as_str()) {
                        tokens.push(text.to_owned());
                    }
                    Ok(())
                },
            )
            .expect("malformed tokens must not poison the live request");
        assert!(
            tokens.is_empty(),
            "malformed tokens must not reach the callback: {tokens:?}"
        );
        assert_eq!(
            done.get("finish_reason").and_then(|v| v.as_str()),
            Some("stop")
        );
        drop(engine);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    fn dummy_child() -> Child {
        // A system executable avoids racing a prior test process that still
        // has a generated script open, which can make spawn fail with ETXTBSY.
        Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn dummy")
    }

    #[cfg(not(unix))]
    fn dummy_child() -> Child {
        Command::new("cmd")
            .args(["/C", "pause"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn dummy")
    }

    fn dummy_stdin() -> ChildStdin {
        // Take from a short-lived child; Engine tests that only call
        // validate_reset_ack never write.
        let mut child = Command::new("true")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("true");
        child.stdin.take().expect("stdin")
    }

    fn dummy_stdout() -> std::process::ChildStdout {
        let mut child = Command::new("true")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("true");
        child.stdout.take().expect("stdout")
    }
}
