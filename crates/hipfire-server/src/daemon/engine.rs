use std::path::{Path, PathBuf};
use std::process::Stdio;

use futures::future::BoxFuture;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tracing::debug;

use super::protocol::{DaemonRequest, DaemonResponse};

trait DaemonTransport: Send {
    fn send_json<'a>(
        &'a mut self,
        req: &'a DaemonRequest,
    ) -> BoxFuture<'a, anyhow::Result<()>>;
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

    /// Send `load` and wait for `loaded`.
    pub async fn load(
        &mut self,
        model_path: &str,
        params: super::protocol::LoadParams,
    ) -> anyhow::Result<super::protocol::LoadedResponse> {
        let request_id = uuid::Uuid::new_v4().to_string();
        self.send(&DaemonRequest::Load(super::protocol::LoadRequest {
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
                                r.response_id, expected_response
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

    /// Send `generate` and collect all tokens. Returns (text, done).
    pub async fn generate(
        &mut self,
        req: super::protocol::GenerateRequest,
    ) -> anyhow::Result<(String, super::protocol::DoneResponse)> {
        let request_id = req.id.clone();
        self.send(&DaemonRequest::Generate(req)).await?;
        let mut text = String::new();
        loop {
            match self.recv().await? {
                DaemonResponse::Token(t) => {
                    if t.id == request_id {
                        text.push_str(&t.text)
                    }
                }
                DaemonResponse::Done(d) => {
                    if d.id == request_id {
                        return Ok((text, d));
                    }
                    tracing::warn!("stale done response: got id={} expected={}", d.id, request_id);
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
        req: super::protocol::GenerateRequest,
        mut on_token: F,
    ) -> anyhow::Result<super::protocol::DoneResponse>
    where
        F: FnMut(String),
    {
        let request_id = req.id.clone();
        self.send(&DaemonRequest::Generate(req)).await?;
        loop {
            match self.recv().await? {
                DaemonResponse::Token(t) => {
                    if t.id == request_id {
                        on_token(t.text)
                    }
                }
                DaemonResponse::Done(d) => {
                    if d.id == request_id {
                        return Ok(d);
                    }
                    tracing::warn!("stale done response: got id={} expected={}", d.id, request_id);
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
/// 3. `./target/release/hipfire-daemon`
/// 4. `./target/debug/hipfire-daemon`
pub fn find_daemon_bin() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HIPFIRE_DAEMON_BIN") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }

    if let Some(home) = dirs::home_dir() {
        let hipfire_bin = home.join(".hipfire").join("bin");
        for name in &["hipfire-daemon", "daemon"] {
            let p = hipfire_bin.join(name);
            if p.exists() {
                return Some(p);
            }
        }
    }

    for rel in &[
        "target/release/hipfire-daemon",
        "target/debug/hipfire-daemon",
    ] {
        let p = PathBuf::from(rel);
        if p.exists() {
            return Some(p);
        }
    }

    None
}
