// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Async daemon JSONL process adapter.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use futures::future::BoxFuture;
use hipfire_daemon_protocol::{
    DaemonRequest, DaemonResponse, DoneResponse, GenerateRequest, LoadParams, LoadRequest,
    LoadedResponse,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tracing::debug;

trait DaemonTransport: Send {
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
        params: LoadParams,
    ) -> anyhow::Result<LoadedResponse> {
        let request_id = uuid::Uuid::new_v4().to_string();
        self.send(&DaemonRequest::Load(LoadRequest {
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
        req: GenerateRequest,
    ) -> anyhow::Result<(String, DoneResponse)> {
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
        req: GenerateRequest,
        mut on_token: F,
    ) -> anyhow::Result<DoneResponse>
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

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_daemon_protocol::GenerationSamplingPolicy;
    use std::collections::VecDeque;

    struct MockTransport {
        sent: Vec<String>,
        responses: VecDeque<DaemonResponse>,
    }

    impl DaemonTransport for MockTransport {
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

    #[tokio::test]
    async fn load_ignores_stale_response_id_and_records_worker() {
        let mut engine = mock_engine(vec![
            DaemonResponse::Loaded(LoadedResponse {
                worker_key_id: "stale-worker".to_string(),
                arch: None,
                dim: None,
                layers: None,
                vocab: None,
                model_worker: None,
                response_id: Some("stale".to_string()),
            }),
            DaemonResponse::Loaded(LoadedResponse {
                worker_key_id: "worker-a".to_string(),
                arch: Some("qwen35".to_string()),
                dim: Some(4096),
                layers: Some(32),
                vocab: Some(151936),
                model_worker: None,
                response_id: None,
            }),
        ]);

        let loaded = engine
            .load("model.hfq", LoadParams::default())
            .await
            .unwrap();
        assert_eq!(loaded.worker_key_id, "worker-a");
        assert_eq!(engine.worker_key_id.as_deref(), Some("worker-a"));
    }

    #[tokio::test]
    async fn generate_collects_only_matching_tokens_until_matching_done() {
        let mut engine = mock_engine(vec![
            DaemonResponse::Token(hipfire_daemon_protocol::TokenResponse {
                id: "other".to_string(),
                text: "skip".to_string(),
            }),
            DaemonResponse::Token(hipfire_daemon_protocol::TokenResponse {
                id: "req-1".to_string(),
                text: "hello".to_string(),
            }),
            DaemonResponse::Token(hipfire_daemon_protocol::TokenResponse {
                id: "req-1".to_string(),
                text: " world".to_string(),
            }),
            DaemonResponse::Done(DoneResponse {
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

        let req = GenerateRequest {
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
            thinking: None,
            max_think_tokens: None,
            request_id: None,
        };
        let (text, done) = engine.generate(req).await.unwrap();
        assert_eq!(text, "hello world");
        assert_eq!(done.tokens, 2);
    }
}
