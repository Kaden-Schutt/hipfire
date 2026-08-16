// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! OpenAI HTTP gateway.
//!
//! Concern: request routing, JSON body limits, streaming vs non-streaming
//! framing, SSE acknowledgement, CORS/health endpoints. Isolates `tiny_http`
//! I/O from business logic.

use anyhow::{bail, Context, Result};
use std::{io::{Read, Write}, sync::{Arc, mpsc}, thread, time::Duration};
use tiny_http::{Header, Method, Request, Response, StatusCode};
use crate::serve::{ServeShared, ServeMeta, is_batch_eligible_request};
use crate::serve::complete::{self, Completion, complete_request, gate_chat_completions_tools, openai_stream_delta_for_event, openai_stream_terminal_chunks, completion_json};
use crate::serve::{Admission, AdmissionGuard, AdmissionError};
use crate::{list_local_models, unix_timestamp};

pub(crate) fn handle_http(mut request: Request, shared: Arc<ServeShared>) -> Result<()> {
    let path = request
        .url()
        .split('?')
        .next()
        .unwrap_or(request.url())
        .to_owned();
    match (request.method(), path.as_str()) {
        (&Method::Get, "/health") => {
            let meta = shared
                .meta
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            request.respond(json_response(
                serde_json::json!({
                    "status": "ok",
                    "model": meta.current_model,
                    "loading_model": meta.loading_model,
                    "pid": std::process::id(),
                    "token": meta.instance_token,
                    "native": true,
                }),
                200,
            ))?;
        }
        (&Method::Get, "/stats") => {
            let meta = shared
                .meta
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            request.respond(json_response(
                serde_json::json!({
                    "model": meta.current_model,
                    "uptime_sec": meta.started.elapsed().as_secs(),
                    "queue_depth": shared.admission.inflight(),
                    "requests_served": meta.requests_served,
                    "retries_attempted": meta.retries_attempted,
                    "retries_succeeded": meta.retries_succeeded,
                    "recent_tok_s": meta.recent_tok_s,
                }),
                200,
            ))?;
        }
        (&Method::Get, "/v1/models") => {
            let runtime = shared
                .runtime
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let local = list_local_models(&runtime.paths, &runtime.registry)?;
            request.respond(json_response(
                serde_json::json!({
                    "object": "list",
                    "data": local.into_iter().map(|model| serde_json::json!({
                        "id": model.registry_tag.unwrap_or(model.name),
                        "object": "model",
                        "owned_by": "hipfire",
                    })).collect::<Vec<_>>()
                }),
                200,
            ))?;
        }
        (&Method::Options, _) => {
            request.respond(
                Response::empty(204)
                    .with_header(header("Access-Control-Allow-Origin", "*"))
                    .with_header(header(
                        "Access-Control-Allow-Headers",
                        "Content-Type, Authorization",
                    ))
                    .with_header(header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")),
            )?;
        }
        (&Method::Post, "/v1/chat/completions") => {
            let body = match read_request_json(&mut request, shared.max_request_bytes) {
                Ok(body) => body,
                Err(error) => {
                    let message = error.to_string();
                    let status = if message.contains("exceeds") {
                        413
                    } else {
                        400
                    };
                    request.respond(openai_error(&message, status))?;
                    return Ok(());
                }
            };
            // Class-aware admission: eligible requests share capacity up to
            // continuous_batch_size, ineligible are exclusive single-flight.
            let (is_eligible, model_for_lease) = {
                let runtime = shared
                    .runtime
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                let tp = runtime.tp;
                let arch = runtime.current_arch.clone();
                let batch_capable = runtime.continuous_batch_capable;
                drop(runtime);
                let eligible = is_batch_eligible_request(&body, tp, arch.as_deref(), batch_capable);
                let model = body
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned());
                (eligible, model)
            };
            let guard = if is_eligible {
                match shared
                    .admission
                    .acquire_for(true, model_for_lease.as_deref())
                {
                    Ok(g) => g,
                    Err(e) => {
                        request.respond(admission_error_response(&e))?;
                        return Ok(());
                    }
                }
            } else {
                match shared.admission.acquire() {
                    Ok(g) => g,
                    Err(e) => {
                        request.respond(admission_error_response(&e))?;
                        return Ok(());
                    }
                }
            };
            // Tools require a lossless endpoint adapter before any generation.
            if let Err(error) = gate_chat_completions_tools(&body) {
                request.respond(openai_error(&error.to_string(), 400))?;
                return Ok(());
            }
            if body.get("stream").and_then(serde_json::Value::as_bool) == Some(true) {
                respond_streaming(request, shared, body, guard)?;
            } else {
                respond_nonstreaming(request, shared, body, guard)?;
            }
        }
        _ => request.respond(openai_error("not found", 404))?,
    }
    Ok(())
}

pub(crate) fn request_error_status(message: &str) -> u16 {
    let lower = message.to_ascii_lowercase();
    if lower.contains("model not found") {
        404
    } else if lower.contains("kv budget")
        || lower.contains("max_tokens")
        || lower.contains("invalid")
        || lower.contains("required")
        || lower.contains("endpoint adapter")
        || lower.contains("lossy")
        || lower.contains("malformed canonical tool call")
    {
        400
    } else {
        500
    }
}

pub(crate) fn read_request_json(request: &mut Request, max_bytes: u64) -> Result<serde_json::Value> {
    if request
        .headers()
        .iter()
        .find(|header| header.field.equiv("Content-Length"))
        .and_then(|header| header.value.as_str().parse::<u64>().ok())
        .is_some_and(|length| length > max_bytes)
    {
        bail!("request body exceeds {max_bytes} bytes");
    }
    let mut bytes = Vec::new();
    request
        .as_reader()
        .take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes {
        bail!("request body exceeds {max_bytes} bytes");
    }
    serde_json::from_slice(&bytes).context("request body is not valid JSON")
}

pub(crate) fn respond_streaming(
    request: Request,
    shared: Arc<ServeShared>,
    body: serde_json::Value,
    guard: AdmissionGuard,
) -> Result<()> {
    let (sender, receiver) = mpsc::channel::<ResponseChunk>();
    thread::spawn(move || {
        let id = request_id();
        let created = unix_timestamp();
        let include_usage = body
            .pointer("/stream_options/include_usage")
            .and_then(serde_json::Value::as_bool)
            == Some(true);
        let model = body
            .get("model")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let first = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": { "role": "assistant" }, "finish_reason": null }],
        });
        let _ = sender.send(ResponseChunk::plain(sse_data(&first)));
        let result = complete_request(
            &shared,
            &body,
            guard,
            Some((id.clone(), created)),
            |event| forward_sse_stream_event(&sender, &id, created, &model, event),
            |completion| {
                // Full terminal representation before Engine can commit.
                deliver_sse_terminal_ack(&sender, completion, include_usage)
            },
        );
        finish_sse_stream(sender, result);
    });
    let mut writer = request.into_writer();
    // Write status line + headers manually. We own the socket, so use
    // Connection: close and close after the terminal chunk; do not emit
    // keep-alive which we would then violate.
    let header_bytes = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    if writer.write_all(header_bytes).is_err() || writer.flush().is_err() {
        // Client disconnected before headers — fail any queued ack and stop.
        // Receiver will be dropped; sender's ack waiters see channel close as Cancelled.
        return Ok(());
    }
    // Any write failure here means the client is gone; the shape of the error
    // does not change what we do, so it is not inspected.
    loop {
        let chunk = match receiver.recv() {
            Ok(c) => c,
            Err(_) => break,
        };
        if chunk.fail {
            if let Some(ack) = chunk.ack {
                let _ = ack.send(Err(()));
            }
            // Unclean failure — abort; HTTP terminator still sent below
            // so the client sees a clean HTTP EOF with incomplete SSE.
            break;
        }
        if chunk.bytes.is_empty() {
            // Empty chunks carry no wire bytes. Per contract never fire an ack for
            // an empty chunk — drop it without sending.
            drop(chunk.ack);
            continue;
        }
        // Framed chunk: "{len:x}\r\n" + payload + "\r\n", then flush.
        let len_hex = format!("{:x}\r\n", chunk.bytes.len());
        let write_res = (|| -> std::io::Result<()> {
            writer.write_all(len_hex.as_bytes())?;
            writer.write_all(&chunk.bytes)?;
            writer.write_all(b"\r\n")?;
            writer.flush()?;
            Ok(())
        })();
        match write_res {
            Ok(()) => {
                if let Some(ack) = chunk.ack {
                    let _ = ack.send(Ok(()));
                }
            }
            Err(_) => {
                if let Some(ack) = chunk.ack {
                    let _ = ack.send(Err(()));
                }
                break;
            }
        }
    }
    // Always send the HTTP chunked terminator so the HTTP body is
    // considered complete. For clean close this is the normal end;
    // for fail (premature EOF) the terminator makes the HTTP layer
    // succeed, letting `read_openai_sse` see an SSE EOF without
    // finish/DONE and return `PrematureEof` (the expected test shape)
    // rather than a lower-level `Io(UnexpectedEof)`.
    let _ = writer.write_all(b"0\r\n\r\n");
    let _ = writer.flush();
    // Dropping the writer closes the socket, which is what `Connection: close`
    // promised; the client then sees EOF.
    drop(writer);
    Ok(())
}

/// Non-stream OpenAI completion: stage the full JSON body before commit, then
/// wait for worker commit+done before EOF. Pre-terminal failures keep error status.
pub(crate) fn respond_nonstreaming(
    request: Request,
    shared: Arc<ServeShared>,
    body: serde_json::Value,
    guard: AdmissionGuard,
) -> Result<()> {
    let (sender, receiver) = mpsc::channel::<ResponseChunk>();
    let (status_tx, status_rx) = mpsc::channel::<Result<(), String>>();
    thread::spawn(move || {
        let result = complete_request(
            &shared,
            &body,
            guard,
            None,
            |_event| Ok(()),
            |completion| {
                let bytes = serde_json::to_vec(&completion_json(completion)).map_err(|err| {
                    hipfire_client::ClientError::Protocol(format!(
                        "completion json serialize failed: {err}"
                    ))
                })?;
                if bytes.is_empty() {
                    return Err(hipfire_client::ClientError::Protocol(
                        "nonstream terminal body must be non-empty".into(),
                    ));
                }
                let (ack_tx, ack_rx) = mpsc::channel();
                sender
                    .send(ResponseChunk {
                        bytes,
                        ack: Some(ack_tx),
                        fail: false,
                    })
                    .map_err(|_| hipfire_client::ClientError::Cancelled)?;
                // Signal handler that terminal bytes are staged (success headers).
                let _ = status_tx.send(Ok(()));
                match ack_rx.recv() {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(_)) | Err(_) => Err(hipfire_client::ClientError::Cancelled),
                }
            },
        );
        match result {
            Ok(_completion) => {
                // Terminal already delivered+acked; close body with no post-commit bytes.
                drop(sender);
            }
            Err(error) => {
                let cancelled = error
                    .downcast_ref::<hipfire_client::ClientError>()
                    .is_some_and(|err| matches!(err, hipfire_client::ClientError::Cancelled));
                if cancelled {
                    // Drop without framing — unclean only if bytes already went out.
                    drop(sender);
                    return;
                }
                // If terminal was never staged, report error status to the handler.
                let message = error.to_string();
                if status_tx.send(Err(message)).is_err() {
                    // Handler already started success body — force unclean close.
                    drop(sender);
                }
            }
        }
    });

    match status_rx.recv() {
        Ok(Ok(())) => {
            // Terminal body staged — success headers, reader owns JSON + waits for EOF.
            request.respond(Response::new(
                StatusCode(200),
                vec![
                    header("Content-Type", "application/json"),
                    header("Access-Control-Allow-Origin", "*"),
                ],
                ChannelReader::new(receiver),
                None,
                None,
            ))?;
        }
        Ok(Err(message)) => {
            request.respond(openai_error(&message, request_error_status(&message)))?;
        }
        Err(_) => {
            // Worker died before status — treat as internal failure.
            request.respond(openai_error("generation worker disconnected", 500))?;
        }
    }
    Ok(())
}

pub(crate) fn request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!(
        "chatcmpl-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

pub(crate) fn sse_data(value: &serde_json::Value) -> Vec<u8> {
    format!("data: {}\n\n", value).into_bytes()
}

/// Forward one logical generate event onto the OpenAI SSE channel.
///
/// Delta-bearing events serialize to plain (no-ack) SSE bytes. No-delta mid-stream
/// events (e.g. withheld tool_calls) are silent — terminal ack handles pure-tool
/// delivery. A dropped receiver maps to [`hipfire_client::ClientError::Cancelled`].
pub(crate) fn forward_sse_stream_event(
    sender: &mpsc::Sender<ResponseChunk>,
    id: &str,
    created: u64,
    model: &str,
    event: &serde_json::Value,
) -> Result<(), hipfire_client::ClientError> {
    if let Some(delta) = openai_stream_delta_for_event(event) {
        let chunk = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
        });
        sender
            .send(ResponseChunk::plain(sse_data(&chunk)))
            .map_err(|_| hipfire_client::ClientError::Cancelled)
    } else {
        // Mid-stream no-delta: do not queue empty probes. Terminal path acks.
        let _ = sender;
        Ok(())
    }
}

/// Serialize terminal tool_calls (if safe), finish, optional usage, and `[DONE]`
/// into one non-empty acknowledged chunk. Waits for ChannelReader progress ack.
pub(crate) fn deliver_sse_terminal_ack(
    sender: &mpsc::Sender<ResponseChunk>,
    completion: &Completion,
    include_usage: bool,
) -> Result<(), hipfire_client::ClientError> {
    let mut bytes = Vec::new();
    for chunk in openai_stream_terminal_chunks(completion, include_usage) {
        bytes.extend_from_slice(&sse_data(&chunk));
    }
    bytes.extend_from_slice(b"data: [DONE]\n\n");
    if bytes.is_empty() {
        return Err(hipfire_client::ClientError::Protocol(
            "stream terminal payload must be non-empty".into(),
        ));
    }
    let (ack_tx, ack_rx) = mpsc::channel();
    sender
        .send(ResponseChunk {
            bytes,
            ack: Some(ack_tx),
            fail: false,
        })
        .map_err(|_| hipfire_client::ClientError::Cancelled)?;
    match ack_rx.recv() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(_)) | Err(_) => Err(hipfire_client::ClientError::Cancelled),
    }
}

/// Close an OpenAI SSE body after `complete_request`.
///
/// Success: terminal already delivered+acked at commit_ready — emit no post-commit
/// bytes. Cancelled: no server_error/`[DONE]`. Post-terminal engine errors force
/// an unclean reader failure rather than appending a success/error frame.
pub(crate) fn finish_sse_stream(sender: mpsc::Sender<ResponseChunk>, result: Result<Completion>) {
    match result {
        Ok(_completion) => {
            // Terminal representation already went out before commit.
            drop(sender);
        }
        Err(error) => {
            let cancelled = error
                .downcast_ref::<hipfire_client::ClientError>()
                .is_some_and(|err| matches!(err, hipfire_client::ClientError::Cancelled));
            if cancelled {
                drop(sender);
                return;
            }
            eprintln!("[hipfire] streaming completion failed: {error:#}");
            // Unclean failure: poison the reader instead of framing success/error.
            let _ = sender.send(ResponseChunk {
                bytes: Vec::new(),
                ack: None,
                fail: false,
            });
            // Marker for reader: empty+no-ack is ignored; use fail signal via drop
            // after a special poison is not needed — ChannelReader fails when the
            // optional fail flag is set. Prefer ResponseChunk::fail.
            let _ = sender.send(ResponseChunk::fail());
            drop(sender);
        }
    }
}

pub(crate) fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("static HTTP header")
}

pub(crate) fn json_response(value: serde_json::Value, status: u16) -> Response<std::io::Cursor<Vec<u8>>> {
    let bytes = serde_json::to_vec(&value).expect("JSON value serializes");
    Response::new(
        StatusCode(status),
        vec![
            header("Content-Type", "application/json"),
            header("Access-Control-Allow-Origin", "*"),
        ],
        std::io::Cursor::new(bytes.clone()),
        Some(bytes.len()),
        None,
    )
}

pub(crate) fn openai_error(message: &str, status: u16) -> Response<std::io::Cursor<Vec<u8>>> {
    let error_type = if (400..500).contains(&status) {
        "invalid_request_error"
    } else {
        "server_error"
    };
    json_response(
        serde_json::json!({
            "error": { "message": message, "type": error_type }
        }),
        status,
    )
}

pub(crate) fn admission_error_response(error: &AdmissionError) -> Response<std::io::Cursor<Vec<u8>>> {
    openai_error(&error.message, 503).with_header(header(
        "Retry-After",
        &error.retry_after_seconds.to_string(),
    ))
}

/// One HTTP response body record. Optional `ack` is signaled only after the
/// reader fully drains `bytes` and the *next* `read` begins (proving writer
/// progress). Queue insertion alone never acknowledges. Drop before that next
/// read disconnects the waiter as Cancelled.
#[derive(Debug)]
pub(crate) struct ResponseChunk {
    bytes: Vec<u8>,
    ack: Option<mpsc::Sender<Result<(), ()>>>,
    /// When set, the next read fails uncleanly (post-terminal engine error).
    fail: bool,
}

impl ResponseChunk {
    pub(crate) fn plain(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            ack: None,
            fail: false,
        }
    }

    pub(crate) fn fail() -> Self {
        Self {
            bytes: Vec::new(),
            ack: None,
            fail: true,
        }
    }
}

pub(crate) struct ChannelReader {
    receiver: mpsc::Receiver<ResponseChunk>,
    current: std::io::Cursor<Vec<u8>>,
    /// Ack to fire on the *next* read after the current chunk is fully drained.
    pending_ack: Option<mpsc::Sender<Result<(), ()>>>,
    failed: bool,
}

impl ChannelReader {
    pub(crate) fn new(receiver: mpsc::Receiver<ResponseChunk>) -> Self {
        Self {
            receiver,
            current: std::io::Cursor::new(Vec::new()),
            pending_ack: None,
            failed: false,
        }
    }

    pub(crate) fn fire_pending_ack(&mut self) {
        if let Some(ack) = self.pending_ack.take() {
            let _ = ack.send(Ok(()));
        }
    }
}

impl Drop for ChannelReader {
    fn drop(&mut self) {
        // Drop before the next-read ack → waiter sees disconnect/Cancelled.
        if let Some(ack) = self.pending_ack.take() {
            let _ = ack.send(Err(()));
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        if self.failed {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "response body failed after terminal delivery",
            ));
        }
        // Next-read after full drain acknowledges the prior chunk. Partial
        // reads must keep draining without firing the pending ack.
        if self.current.position() == self.current.get_ref().len() as u64 {
            self.fire_pending_ack();
        }

        loop {
            let read = self.current.read(output)?;
            if read > 0 {
                return Ok(read);
            }
            // Current buffer exhausted. Do not ack yet — ack waits for *next* read.
            match self.receiver.recv() {
                Ok(chunk) if chunk.fail => {
                    self.failed = true;
                    if let Some(ack) = chunk.ack {
                        let _ = ack.send(Err(()));
                    }
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "response body failed after terminal delivery",
                    ));
                }
                // Empty non-fail chunks are ignored (no ack on empty).
                Ok(chunk) if chunk.bytes.is_empty() => {
                    if let Some(ack) = chunk.ack {
                        // Empty acknowledged chunk is invalid — disconnect waiter.
                        let _ = ack.send(Err(()));
                    }
                    continue;
                }
                Ok(chunk) => {
                    // If a previous chunk still had a pending ack (shouldn't with
                    // single outstanding), fire it only on this next read entry —
                    // already fired at top. Stage this chunk's ack for the read
                    // *after* it is fully drained.
                    self.current = std::io::Cursor::new(chunk.bytes);
                    self.pending_ack = chunk.ack;
                }
                Err(_) => {
                    // Channel closed: any pending ack is a disconnect.
                    if let Some(ack) = self.pending_ack.take() {
                        let _ = ack.send(Err(()));
                    }
                    return Ok(0);
                }
            }
        }
    }
}





