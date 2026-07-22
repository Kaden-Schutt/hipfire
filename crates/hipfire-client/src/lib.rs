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
    collections::BTreeMap,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    time::Duration,
};
use thiserror::Error;

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
    #[error("HTTP service error: {0}")]
    Http(String),
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

/// Stream an OpenAI-compatible chat response. The callback receives visible
/// reasoning and answer content in wire order. Returning true from `cancelled`
/// stops reading after the current SSE event.
pub fn stream_openai_chat(
    host: &str,
    port: u16,
    mut body: Value,
    timeout: Duration,
    mut delta: impl FnMut(&str) -> Result<()>,
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
    for line in reader.lines() {
        if cancelled() {
            return Ok(());
        }
        let line = line?;
        let Some(payload) = line.trim().strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if payload == "[DONE]" {
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
        let Some(delta_value) = value
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("delta"))
        else {
            continue;
        };
        for field in ["reasoning_content", "content"] {
            if let Some(text) = delta_value.get(field).and_then(Value::as_str) {
                if !text.is_empty() {
                    delta(text)?;
                }
            }
        }
    }
    Ok(())
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

pub struct Engine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
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
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
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
        let visibility = hipfire_config::synchronized_device_visibility(
            config,
            hip.as_deref(),
            rocr.as_deref(),
        )
        .map_err(|error| ClientError::Protocol(error.to_string()))?;
        if let Some(visibility) = visibility {
            environment.insert(hipfire_config::HIP_VISIBLE_DEVICES.into(), visibility.hip);
            environment.insert(hipfire_config::ROCR_VISIBLE_DEVICES.into(), visibility.rocr);
        }

        let mut engine = Self::spawn(daemon, &environment)?;
        let response = engine.request(&serde_json::json!({
            "type": "configure",
            "config": config,
        }))?;
        expect_type(&response, "configured")?;
        Ok(engine)
    }

    pub fn send(&mut self, message: &Value) -> Result<()> {
        serde_json::to_writer(&mut self.stdin, message).map_err(|error| {
            ClientError::Protocol(format!("failed to serialize request: {error}"))
        })?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        Ok(())
    }

    pub fn recv(&mut self) -> Result<Value> {
        let mut line = String::new();
        loop {
            line.clear();
            if self.stdout.read_line(&mut line)? == 0 {
                let status = self
                    .child
                    .try_wait()
                    .ok()
                    .flatten()
                    .map(|status| status.to_string())
                    .unwrap_or_else(|| "unknown".into());
                return Err(ClientError::Closed { status });
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            return serde_json::from_str(trimmed).map_err(|error| ClientError::InvalidJson {
                message: error.to_string(),
                line: trimmed.chars().take(512).collect(),
            });
        }
    }

    pub fn request(&mut self, message: &Value) -> Result<Value> {
        self.send(message)?;
        self.recv()
    }

    pub fn ping(&mut self) -> Result<()> {
        let response = self.request(&serde_json::json!({ "type": "ping" }))?;
        expect_type(&response, "pong")
    }

    pub fn load(&mut self, model: &Path, params: Value) -> Result<Value> {
        let response = self.request(&serde_json::json!({
            "type": "load",
            "model": model,
            "params": params,
        }))?;
        if response.get("type").and_then(Value::as_str) == Some("error") {
            return Err(ClientError::Protocol(error_message(&response)));
        }
        // Optional PFlash/info events can precede the terminal loaded event.
        let mut response = response;
        while response.get("type").and_then(Value::as_str) != Some("loaded") {
            response = self.recv()?;
            if response.get("type").and_then(Value::as_str) == Some("error") {
                return Err(ClientError::Protocol(error_message(&response)));
            }
        }
        Ok(response)
    }

    pub fn generate(
        &mut self,
        request: &Value,
        mut event: impl FnMut(&Value) -> Result<()>,
    ) -> Result<Value> {
        self.send(request)?;
        loop {
            let value = self.recv()?;
            event(&value)?;
            match value.get("type").and_then(Value::as_str) {
                Some("done") => return Ok(value),
                Some("error") => return Err(ClientError::Protocol(error_message(&value))),
                _ => {}
            }
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        let response = self.request(&serde_json::json!({ "type": "reset" }))?;
        expect_type(&response, "reset")
    }

    pub fn unload(&mut self) -> Result<()> {
        let response = self.request(&serde_json::json!({ "type": "unload" }))?;
        expect_type(&response, "unloaded")
    }

    pub fn child_id(&self) -> u32 {
        self.child.id()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.send(&serde_json::json!({ "type": "unload" }));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn expect_type(response: &Value, expected: &str) -> Result<()> {
    match response.get("type").and_then(Value::as_str) {
        Some(actual) if actual == expected => Ok(()),
        Some("error") => Err(ClientError::Protocol(error_message(response))),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, fs};

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
        use std::os::unix::fs::PermissionsExt;
        let root = env::temp_dir().join(format!("hipfire-client-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let daemon = root.join("daemon");
        fs::write(
            &daemon,
            "#!/bin/sh\nwhile IFS= read -r line; do\n case \"$line\" in *'\"ping\"'*) echo '{\"type\":\"pong\"}' ;; *'\"unload\"'*) echo '{\"type\":\"unloaded\"}'; exit 0 ;; esac\ndone\n",
        )
        .unwrap();
        fs::set_permissions(&daemon, fs::Permissions::from_mode(0o755)).unwrap();
        let mut engine = Engine::spawn(&daemon, &BTreeMap::new()).unwrap();
        engine.ping().unwrap();
        engine.unload().unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn configured_spawn_sends_process_policy_before_ping() {
        use std::os::unix::fs::PermissionsExt;
        let root = env::temp_dir().join(format!(
            "hipfire-client-configured-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let daemon = root.join("daemon");
        fs::write(
            &daemon,
            "#!/bin/sh\nconfigured=0\nwhile IFS= read -r line; do\n case \"$line\" in *'\"configure\"'*) if [ \"$HIP_VISIBLE_DEVICES\" = '0,1' ] && [ \"$ROCR_VISIBLE_DEVICES\" = '2,3' ]; then configured=1; echo '{\"type\":\"configured\"}'; else echo '{\"type\":\"error\",\"message\":\"device visibility is not synchronized\"}'; fi ;; *'\"ping\"'*) if [ \"$configured\" = 1 ]; then echo '{\"type\":\"pong\"}'; else echo '{\"type\":\"error\",\"message\":\"not configured\"}'; fi ;; *'\"unload\"'*) echo '{\"type\":\"unloaded\"}'; exit 0 ;; esac\ndone\n",
        )
        .unwrap();
        fs::set_permissions(&daemon, fs::Permissions::from_mode(0o755)).unwrap();
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
        let mut engine = Engine::spawn_configured(&daemon, &BTreeMap::new(), &config).unwrap();
        engine.ping().unwrap();
        engine.unload().unwrap();
        let _ = fs::remove_dir_all(root);
    }
}
