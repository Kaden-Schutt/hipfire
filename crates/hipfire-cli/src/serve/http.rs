// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! OpenAI HTTP gateway.
//!
//! Concern: request routing, JSON body limits, streaming vs non-streaming
//! framing, SSE acknowledgement, CORS/health endpoints. Isolates Hyper
//! I/O from business logic.

use crate::serve::complete::{
    complete_request_cancellable, completion_json, gate_chat_completions_tools,
    openai_stream_delta_for_event, openai_stream_terminal_chunks, Completion,
};
use crate::serve::{is_batch_eligible_request, ServeShared};
use crate::serve::{AdmissionError, AdmissionGuard};
use crate::{list_local_models, unix_timestamp};
use anyhow::{bail, Context, Result};
use bytes::Bytes;
use http_body_util::{combinators::UnsyncBoxBody, BodyExt, Full};
use hyper::server::conn::http1;
use hyper::{
    body::{Frame, Incoming},
    header, Method, Request, Response,
};
use hyper_util::rt::TokioIo;
use std::{
    convert::Infallible,
    io,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    task::{Context as TaskContext, Poll},
};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Boxed body and helpers
// ---------------------------------------------------------------------------

pub(crate) type BoxBody = UnsyncBoxBody<Bytes, std::io::Error>;

fn boxed<B>(body: B) -> BoxBody
where
    B: hyper::body::Body<Data = Bytes, Error = std::io::Error> + Send + 'static,
{
    UnsyncBoxBody::new(body)
}

fn boxed_full(bytes: Vec<u8>) -> BoxBody {
    boxed(Full::new(bytes.into()).map_err(|never: Infallible| match never {}))
}

fn boxed_empty() -> BoxBody {
    boxed(Full::new(Bytes::new()).map_err(|never: Infallible| match never {}))
}

pub(crate) fn json_response(value: serde_json::Value, status: u16) -> Response<BoxBody> {
    match json_response_result(&value, status) {
        Ok(resp) => resp,
        Err(message) => openai_error(&message, 500),
    }
}

fn json_response_result(value: &serde_json::Value, status: u16) -> Result<Response<BoxBody>, String> {
    let bytes = serde_json::to_vec(value).map_err(|err| format!("failed to encode JSON response: {err}"))?;
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(boxed_full(bytes))
        .map_err(|err| format!("failed to build HTTP response: {err}"))
}

/// Last-resort 500 body with no serde dependency, so error rendering always
/// terminates even if JSON encoding itself is what failed.
fn static_server_error() -> Response<BoxBody> {
    Response::builder()
        .status(500)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(boxed_full(
            br#"{"error":{"message":"internal server error","type":"server_error"}}"#.to_vec(),
        ))
        // Static status, headers, and body: the builder cannot fail on these
        // inputs, and there is no further fallback below this point.
        .expect("static 500 response builds")
}

pub(crate) fn openai_error(message: &str, status: u16) -> Response<BoxBody> {
    let error_type = if (400..500).contains(&status) {
        "invalid_request_error"
    } else {
        "server_error"
    };
    json_response_result(
        &serde_json::json!({
            "error": { "message": message, "type": error_type }
        }),
        status,
    )
    .unwrap_or_else(|_| static_server_error())
}

pub(crate) fn admission_error_response(error: &AdmissionError) -> Response<BoxBody> {
    let mut resp = openai_error(&error.message, 503);
    if let Ok(retry_after) =
        header::HeaderValue::from_str(&error.retry_after_seconds.to_string())
    {
        resp.headers_mut().insert(header::RETRY_AFTER, retry_after);
    }
    resp
}

// ---------------------------------------------------------------------------
// FlushAcks / TrackedIo — ack only after socket flush
// ---------------------------------------------------------------------------

type AckSender = std::sync::mpsc::Sender<Result<(), ()>>;

/// Per-connection FIFO of terminal-ack senders. Bodies register when they yield
/// a frame; `TrackedIo` completes them only after a successful `poll_flush`.
#[derive(Clone, Debug, Default)]
struct FlushAcks {
    queue: Arc<Mutex<Vec<AckSender>>>,
}

impl FlushAcks {
    fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn register(&self, ack: AckSender) {
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(ack);
    }

    fn drain_ok(&self) {
        let pending = std::mem::take(&mut *self.queue.lock().unwrap_or_else(|e| e.into_inner()));
        for ack in pending {
            let _ = ack.send(Ok(()));
        }
    }

    fn drain_err(&self) {
        let pending = std::mem::take(&mut *self.queue.lock().unwrap_or_else(|e| e.into_inner()));
        for ack in pending {
            let _ = ack.send(Err(()));
        }
    }
}

/// TcpStream wrapper that ties terminal acks to successful socket flushes.
struct TrackedIo {
    inner: TcpStream,
    acks: FlushAcks,
}

impl TrackedIo {
    fn new(inner: TcpStream, acks: FlushAcks) -> Self {
        Self { inner, acks }
    }
}

impl Drop for TrackedIo {
    fn drop(&mut self) {
        self.acks.drain_err();
    }
}

impl AsyncRead for TrackedIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for TrackedIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let result = Pin::new(&mut self.inner).poll_write(cx, buf);
        if let Poll::Ready(Err(_)) = &result {
            self.acks.drain_err();
        }
        result
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        let result = Pin::new(&mut self.inner).poll_flush(cx);
        match &result {
            Poll::Ready(Ok(())) => self.acks.drain_ok(),
            Poll::Ready(Err(_)) => self.acks.drain_err(),
            Poll::Pending => {}
        }
        result
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        let result = Pin::new(&mut self.inner).poll_shutdown(cx);
        if let Poll::Ready(Err(_)) = &result {
            self.acks.drain_err();
        }
        result
    }
}

// ---------------------------------------------------------------------------
// AckBody / ChannelBody — preserve commit boundary
// ---------------------------------------------------------------------------

/// Terminal JSON body: exactly one frame. Ack is registered with the connection
/// tracker when the frame is yielded; Ok only after `TrackedIo` flushes. Drop
/// before registration sends Err (correlated abort).
pub(crate) struct AckBody {
    data: Option<Bytes>,
    ack: Option<AckSender>,
    tracker: FlushAcks,
}

impl AckBody {
    pub(crate) fn new(bytes: Vec<u8>, ack: AckSender, tracker: FlushAcks) -> Self {
        Self {
            data: Some(bytes.into()),
            ack: Some(ack),
            tracker,
        }
    }
}

impl Drop for AckBody {
    fn drop(&mut self) {
        if let Some(ack) = self.ack.take() {
            let _ = ack.send(Err(()));
        }
    }
}

impl hyper::body::Body for AckBody {
    type Data = Bytes;
    type Error = std::io::Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if let Some(data) = self.data.take() {
            // Yield the frame and hand the ack to the connection tracker. Ok
            // arrives only after a later successful socket flush — never here.
            if let Some(ack) = self.ack.take() {
                self.tracker.register(ack);
            }
            return Poll::Ready(Some(Ok(Frame::data(data))));
        }
        Poll::Ready(None)
    }
}

/// One SSE frame: plain bytes, optional terminal ack sender, or fail marker.
#[derive(Debug)]
pub(crate) struct ResponseChunk {
    bytes: Vec<u8>,
    ack: Option<AckSender>,
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

/// Streaming SSE body: multiple frames via channel. Dropped receiver closes
/// sender and callback returns Cancelled. Terminal chunk ack is registered with
/// the connection tracker when that frame is yielded (not on a later body poll).
/// Owns a clone of the worker cancellation flag; drop (client disconnect after
/// the response is returned) sets it so long silent towers abort promptly.
pub(crate) struct ChannelBody {
    rx: tokio::sync::mpsc::Receiver<ResponseChunk>,
    tracker: FlushAcks,
    cancelled: Arc<AtomicBool>,
    failed: bool,
}

impl ChannelBody {
    pub(crate) fn new(
        rx: tokio::sync::mpsc::Receiver<ResponseChunk>,
        tracker: FlushAcks,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            rx,
            tracker,
            cancelled,
            failed: false,
        }
    }
}

impl Drop for ChannelBody {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

impl hyper::body::Body for ChannelBody {
    type Data = Bytes;
    type Error = std::io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.failed {
            return Poll::Ready(Some(Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "response body failed after terminal delivery",
            ))));
        }

        match Pin::new(&mut self.rx).poll_recv(cx) {
            Poll::Ready(Some(chunk)) => {
                if chunk.fail {
                    self.failed = true;
                    if let Some(ack) = chunk.ack {
                        let _ = ack.send(Err(()));
                    }
                    return Poll::Ready(Some(Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "response body failed after terminal delivery",
                    ))));
                }
                if chunk.bytes.is_empty() {
                    // Empty chunks carry no wire bytes — never fire ack for empty.
                    if let Some(ack) = chunk.ack {
                        let _ = ack.send(Err(()));
                    }
                    // Poll again for next chunk.
                    return self.poll_frame(cx);
                }
                // Register terminal ack with the connection tracker at yield time.
                // Do not ack on a subsequent body poll — only TrackedIo::poll_flush.
                if let Some(ack) = chunk.ack {
                    self.tracker.register(ack);
                }
                Poll::Ready(Some(Ok(Frame::data(Bytes::from(chunk.bytes)))))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

// ---------------------------------------------------------------------------
// Cancellation guard — Hyper drops handler future on client FIN in 0ms
// ---------------------------------------------------------------------------

struct CancelOnDrop {
    token: CancellationToken,
    cancelled: Arc<AtomicBool>,
    armed: bool,
}

impl CancelOnDrop {
    fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            cancelled: Arc::new(AtomicBool::new(false)),
            armed: true,
        }
    }

    fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    fn cancelled(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancelled)
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancelled.store(true, Ordering::SeqCst);
            self.token.cancel();
        }
    }
}

// ---------------------------------------------------------------------------
// Public Hyper entry point
// ---------------------------------------------------------------------------

pub(crate) async fn serve_listener(listener: TcpListener, shared: Arc<ServeShared>) -> Result<()> {
    serve_listener_until(listener, shared, CancellationToken::new()).await
}

/// Accept loop that exits cleanly when `shutdown` is cancelled.
pub(crate) async fn serve_listener_until(
    listener: TcpListener,
    shared: Arc<ServeShared>,
    shutdown: CancellationToken,
) -> Result<()> {
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => return Ok(()),
            accepted = listener.accept() => {
                let (stream, _) = accepted?;
                let shared = Arc::clone(&shared);
                tokio::spawn(async move {
                    // One tracker per connection so pipelined responses share FIFO
                    // flush ordering without global state.
                    let acks = FlushAcks::new();
                    let io = TrackedIo::new(stream, acks.clone());
                    let service = hyper::service::service_fn(move |req: Request<Incoming>| {
                        let shared = Arc::clone(&shared);
                        let acks = acks.clone();
                        async move {
                            Ok::<_, Infallible>(handle_request(req, shared, acks).await)
                        }
                    });
                    if let Err(err) = http1::Builder::new()
                        .serve_connection(TokioIo::new(io), service)
                        .await
                    {
                        // Hyper already logs connection resets; keep quiet for normal close.
                        let msg = err.to_string();
                        if !msg.contains("incomplete") && !msg.contains("reset") {
                            eprintln!("[hipfire] connection error: {err:#}");
                        }
                    }
                });
            }
        }
    }
}

async fn handle_request(
    req: Request<Incoming>,
    shared: Arc<ServeShared>,
    acks: FlushAcks,
) -> Response<BoxBody> {
    let path = req
        .uri()
        .path()
        .split('?')
        .next()
        .unwrap_or(req.uri().path())
        .to_owned();
    let method = req.method().clone();

    match (method, path.as_str()) {
        (Method::GET, "/health") => {
            let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
            let body = serde_json::json!({
                "status": "ok",
                "model": meta.current_model,
                "loading_model": meta.loading_model,
                "pid": std::process::id(),
                "token": meta.instance_token,
                "native": true,
            });
            json_response(body, 200)
        }
        (Method::GET, "/stats") => {
            let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
            let body = serde_json::json!({
                "model": meta.current_model,
                "uptime_sec": meta.started.elapsed().as_secs(),
                "queue_depth": shared.admission.inflight(),
                "requests_served": meta.requests_served,
                "retries_attempted": meta.retries_attempted,
                "retries_succeeded": meta.retries_succeeded,
                "recent_tok_s": meta.recent_tok_s,
            });
            json_response(body, 200)
        }
        (Method::GET, "/metrics") => {
            let (uptime, model) = {
                let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
                (meta.started.elapsed().as_secs(), meta.current_model.clone())
            };
            let body = shared.metrics.render(
                shared.admission.inflight(),
                shared.admission.capacity(),
                uptime,
                model.as_deref(),
            );
            let mut resp = Response::builder()
                .status(200)
                .header(
                    header::CONTENT_TYPE,
                    "text/plain; version=0.0.4; charset=utf-8",
                )
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .body(boxed_full(body.into_bytes()))
                .unwrap();
            resp
        }
        (Method::GET, "/v1/models") => {
            let runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
            let local = match list_local_models(&runtime.paths, &runtime.registry) {
                Ok(m) => m,
                Err(e) => return openai_error(&e.to_string(), 500),
            };
            let body = serde_json::json!({
                "object": "list",
                "data": local.into_iter().map(|model| serde_json::json!({
                    "id": model.registry_tag.unwrap_or(model.name),
                    "object": "model",
                    "owned_by": "hipfire",
                })).collect::<Vec<_>>()
            });
            json_response(body, 200)
        }
        (Method::OPTIONS, _) => {
            let mut resp = Response::builder()
                .status(204)
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .header(
                    header::ACCESS_CONTROL_ALLOW_HEADERS,
                    "Content-Type, Authorization",
                )
                .header(header::ACCESS_CONTROL_ALLOW_METHODS, "GET, POST, OPTIONS")
                .body(boxed_empty())
                .unwrap();
            resp
        }
        (Method::POST, "/v1/chat/completions") => {
            let max_bytes = shared.max_request_bytes;
            if req
                .headers()
                .get(header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|length| length > max_bytes)
            {
                return openai_error(&format!("request body exceeds {max_bytes} bytes"), 413);
            }
            let body_val = match read_json_body(req.into_body(), max_bytes).await {
                Ok(v) => v,
                Err(err) => {
                    let msg = err.to_string();
                    let status = if msg.contains("exceeds") { 413 } else { 400 };
                    return openai_error(&msg, status);
                }
            };

            let (is_eligible, model_for_lease) = {
                let runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
                let tp = runtime.tp;
                let arch = runtime.current_arch.clone();
                let batch_capable = runtime.continuous_batch_capable;
                let multi_slot = runtime.multi_slot_enabled;
                drop(runtime);
                // The admission gate is transport concurrency, not a batch-mode
                // selector. Experimental slots overlap independent requests
                // while remaining separate from ContinuousBatchScheduler.
                let eligible = multi_slot
                    || is_batch_eligible_request(&body_val, tp, arch.as_deref(), batch_capable);
                let model = body_val
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned());
                (eligible, model)
            };

            let mut cancel_guard = CancelOnDrop::new();
            let cancel = cancel_guard.token();
            let cancelled = cancel_guard.cancelled();

            let guard = if is_eligible {
                match shared
                    .admission
                    .acquire_for_async(true, model_for_lease.as_deref(), cancel.clone())
                    .await
                {
                    Ok(g) => g,
                    Err(e) => return admission_error_response(&e),
                }
            } else {
                match shared.admission.acquire_async(cancel.clone()).await {
                    Ok(g) => g,
                    Err(e) => return admission_error_response(&e),
                }
            };

            if let Err(error) = gate_chat_completions_tools(&body_val) {
                return openai_error(&error.to_string(), 400);
            }

            let is_stream = body_val.get("stream").and_then(|v| v.as_bool()) == Some(true);
            let response = if is_stream {
                handle_streaming(shared, body_val, guard, cancelled, acks).await
            } else {
                handle_nonstreaming(shared, body_val, guard, cancelled, acks).await
            };
            cancel_guard.disarm();
            response
        }
        _ => openai_error("not found", 404),
    }
}

async fn read_json_body(body: Incoming, max_bytes: u64) -> Result<serde_json::Value> {
    let mut bytes = Vec::new();
    let mut stream = body;
    while let Some(frame) = stream.frame().await {
        let frame = frame.context("failed to read request body")?;
        if let Some(data) = frame.data_ref() {
            if bytes.len() as u64 + data.len() as u64 > max_bytes {
                bail!("request body exceeds {max_bytes} bytes");
            }
            bytes.extend_from_slice(data);
        }
        if bytes.len() as u64 > max_bytes {
            bail!("request body exceeds {max_bytes} bytes");
        }
    }
    if bytes.len() as u64 > max_bytes {
        bail!("request body exceeds {max_bytes} bytes");
    }
    serde_json::from_slice(&bytes).context("request body is not valid JSON")
}

// ---------------------------------------------------------------------------
// Streaming / Non-streaming handlers
// ---------------------------------------------------------------------------

async fn handle_streaming(
    shared: Arc<ServeShared>,
    body: serde_json::Value,
    guard: AdmissionGuard,
    cancelled: Arc<AtomicBool>,
    acks: FlushAcks,
) -> Response<BoxBody> {
    let (tx, rx) = tokio::sync::mpsc::channel::<ResponseChunk>(32);

    let id = request_id();
    let created = unix_timestamp();
    let include_usage = body
        .pointer("/stream_options/include_usage")
        .and_then(|v| v.as_bool())
        == Some(true);
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_owned();

    let first = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": { "role": "assistant" }, "finish_reason": null }],
    });
    let _ = tx.try_send(ResponseChunk::plain(sse_data(&first)));

    let tx_clone = tx.clone();
    let shared_clone = Arc::clone(&shared);
    let id_clone = id.clone();
    let model_clone = model.clone();
    let body_cancelled = Arc::clone(&cancelled);
    tokio::task::spawn_blocking(move || {
        let result = complete_request_cancellable(
            &shared_clone,
            &body,
            guard,
            Some((id_clone.clone(), created)),
            &cancelled,
            |event| forward_sse_stream_event(&tx_clone, &id_clone, created, &model_clone, event),
            |completion| deliver_sse_terminal_ack(&tx_clone, completion, include_usage),
        );
        finish_sse_stream(tx_clone, result);
    });

    let body = ChannelBody::new(rx, acks, body_cancelled);
    let mut resp = Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(boxed(body))
        .unwrap();
    // Ensure chunked; hyper sets it automatically for streaming bodies.
    resp
}

async fn handle_nonstreaming(
    shared: Arc<ServeShared>,
    body: serde_json::Value,
    guard: AdmissionGuard,
    cancelled: Arc<AtomicBool>,
    acks: FlushAcks,
) -> Response<BoxBody> {
    // Staged terminal channel: Ok((bytes, ack_tx)) for success, Err(msg) for preterminal.
    let (staged_tx, staged_rx) = tokio::sync::oneshot::channel::<
        Result<(Vec<u8>, std::sync::mpsc::Sender<Result<(), ()>>), String>,
    >();
    let staged_tx = Arc::new(Mutex::new(Some(staged_tx)));

    let shared_clone = Arc::clone(&shared);
    let body_for_worker = body;
    let staged_tx_clone = Arc::clone(&staged_tx);
    let staged_tx_for_worker = Arc::clone(&staged_tx);
    tokio::task::spawn_blocking(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let staged_for_terminal = Arc::clone(&staged_tx_for_worker);
            let terminal = |completion: &Completion| {
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
                let (ack_tx, ack_rx) = std::sync::mpsc::channel();
                // Stage exactly one JSON frame and block for ack (RAII guard cancels while awaiting).
                {
                    let mut lock = staged_for_terminal
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    if let Some(tx) = lock.take() {
                        let _ = tx.send(Ok((bytes, ack_tx)));
                    } else {
                        return Err(hipfire_client::ClientError::Cancelled);
                    }
                }
                match ack_rx.recv() {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(_)) | Err(_) => Err(hipfire_client::ClientError::Cancelled),
                }
            };
            complete_request_cancellable(
                &shared_clone,
                &body_for_worker,
                guard,
                None,
                &cancelled,
                |_event| Ok(()),
                terminal,
            )
        }));
        // If terminal was never staged, propagate preterminal error via staged channel.
        let mut lock = staged_tx_clone
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(tx) = lock.take() {
            match outcome {
                Ok(Ok(_completion)) => {
                    // Ok but terminal never called — report as error.
                    let _ = tx.send(Err(
                        "generation completed without a response body".to_string()
                    ));
                }
                Ok(Err(error)) => {
                    let msg = error.to_string();
                    let _ = tx.send(Err(msg));
                }
                Err(payload) => {
                    let detail = payload
                        .downcast_ref::<&str>()
                        .map(|s| (*s).to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "non-string panic payload".to_string());
                    let _ = tx.send(Err(format!("generation worker panicked: {detail}")));
                }
                Ok(Ok(_)) => {}
            }
        }
    });

    // Await staged terminal with handler's CancellationToken guard alive.
    // Hyper dropping this future cancels the token and the AtomicBool watcher above.
    match staged_rx.await {
        Ok(Ok((bytes, ack_tx))) => {
            // Success: exactly one JSON frame with AckBody.
            let body = AckBody::new(bytes, ack_tx, acks);
            let resp = Response::builder()
                .status(200)
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .body(boxed(body))
                .unwrap();
            resp
        }
        Ok(Err(message)) => openai_error(&message, request_error_status(&message)),
        Err(_) => openai_error("generation worker disconnected", 500),
    }
}

// ---------------------------------------------------------------------------
// Preserved business helpers (same shapes, adapted to tokio mpsc where needed)
// ---------------------------------------------------------------------------

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
/// Delta-bearing events serialize to plain (no-ack) SSE bytes. No-delta mid-stream
/// events are silent — terminal ack handles pure-tool delivery. A dropped receiver
/// maps to `Cancelled`.
pub(crate) fn forward_sse_stream_event(
    sender: &tokio::sync::mpsc::Sender<ResponseChunk>,
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
            .blocking_send(ResponseChunk::plain(sse_data(&chunk)))
            .map_err(|_| hipfire_client::ClientError::Cancelled)
    } else {
        let _ = sender;
        Ok(())
    }
}

/// Serialize terminal tool_calls (if safe), finish, optional usage, and `[DONE]`
/// into one non-empty acknowledged chunk. Waits for ChannelBody progress ack.
pub(crate) fn deliver_sse_terminal_ack(
    sender: &tokio::sync::mpsc::Sender<ResponseChunk>,
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
    let (ack_tx, ack_rx) = std::sync::mpsc::channel();
    sender
        .blocking_send(ResponseChunk {
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

/// Close an OpenAI SSE body after `complete_request_cancellable`.
/// Success: terminal already delivered+acked at commit_ready — emit no post-commit bytes.
/// Cancelled: no server_error/`[DONE]`. Post-terminal engine errors force an unclean
/// reader failure rather than appending a success/error frame.
pub(crate) fn finish_sse_stream(
    sender: tokio::sync::mpsc::Sender<ResponseChunk>,
    result: Result<Completion>,
) {
    match result {
        Ok(_completion) => {
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
            let _ = sender.try_send(ResponseChunk::fail());
            drop(sender);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::body::Body;
    use std::future::poll_fn;
    use std::task::Waker;
    use std::time::Duration;

    fn noop_cx() -> TaskContext<'static> {
        TaskContext::from_waker(Waker::noop())
    }

    async fn loopback_pair() -> (TrackedIo, TcpStream, FlushAcks) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        let client = TcpStream::connect(addr).await.expect("connect");
        let (server, _) = listener.accept().await.expect("accept");
        let acks = FlushAcks::new();
        let tracked = TrackedIo::new(server, acks.clone());
        (tracked, client, acks)
    }

    async fn write_all(io: &mut TrackedIo, mut buf: &[u8]) {
        while !buf.is_empty() {
            let n = poll_fn(|cx| Pin::new(&mut *io).poll_write(cx, buf))
                .await
                .expect("write");
            assert!(n > 0, "write made no progress");
            buf = &buf[n..];
        }
    }

    async fn flush(io: &mut TrackedIo) {
        poll_fn(|cx| Pin::new(&mut *io).poll_flush(cx))
            .await
            .expect("flush");
    }

    #[tokio::test]
    async fn ack_body_poll_alone_does_not_ack() {
        let acks = FlushAcks::new();
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let mut body = AckBody::new(b"{}".to_vec(), ack_tx, acks.clone());

        let mut cx = noop_cx();
        // Yield data frame — registers with tracker, must not complete Ok yet.
        match Pin::new(&mut body).poll_frame(&mut cx) {
            Poll::Ready(Some(Ok(frame))) => {
                assert!(frame.data_ref().is_some_and(|d| d.as_ref() == b"{}"));
            }
            other => panic!("expected data frame, got {other:?}"),
        }
        // Second poll is EOF — still no Ok ack.
        match Pin::new(&mut body).poll_frame(&mut cx) {
            Poll::Ready(None) => {}
            other => panic!("expected EOF, got {other:?}"),
        }
        assert!(
            ack_rx.try_recv().is_err(),
            "body poll must not ack before socket flush"
        );

        // Completing the tracker flush path delivers Ok.
        acks.drain_ok();
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Ok(())));
    }

    #[tokio::test]
    async fn channel_body_poll_alone_does_not_ack() {
        let acks = FlushAcks::new();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        tx.try_send(ResponseChunk {
            bytes: b"data: hi\n\n".to_vec(),
            ack: Some(ack_tx),
            fail: false,
        })
        .expect("send chunk");
        drop(tx);

        let cancelled = Arc::new(AtomicBool::new(false));
        let mut body = ChannelBody::new(rx, acks.clone(), Arc::clone(&cancelled));
        let mut cx = noop_cx();
        match Pin::new(&mut body).poll_frame(&mut cx) {
            Poll::Ready(Some(Ok(frame))) => {
                assert!(frame
                    .data_ref()
                    .is_some_and(|d| d.as_ref() == b"data: hi\n\n"));
            }
            other => panic!("expected data frame, got {other:?}"),
        }
        // Further polls (EOF) must not complete the terminal ack.
        match Pin::new(&mut body).poll_frame(&mut cx) {
            Poll::Ready(None) => {}
            other => panic!("expected EOF, got {other:?}"),
        }
        assert!(
            ack_rx.try_recv().is_err(),
            "channel body poll must not ack before socket flush"
        );
        acks.drain_ok();
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Ok(())));
    }

    #[tokio::test]
    async fn channel_body_drop_sets_cancelled() {
        let acks = FlushAcks::new();
        let (_tx, rx) = tokio::sync::mpsc::channel::<ResponseChunk>(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        assert!(!cancelled.load(Ordering::SeqCst), "flag must start false");
        let body = ChannelBody::new(rx, acks, Arc::clone(&cancelled));
        drop(body);
        assert!(
            cancelled.load(Ordering::SeqCst),
            "dropping ChannelBody must set cancelled"
        );
    }

    #[tokio::test]
    async fn tracked_flush_delivers_ok_ack() {
        let (mut tracked, _client, acks) = loopback_pair().await;
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let mut body = AckBody::new(b"ok".to_vec(), ack_tx, acks);

        let mut cx = noop_cx();
        assert!(matches!(
            Pin::new(&mut body).poll_frame(&mut cx),
            Poll::Ready(Some(Ok(_)))
        ));
        assert!(ack_rx.try_recv().is_err(), "no ack before flush");

        write_all(&mut tracked, b"ok").await;
        // Write alone must not ack — only successful poll_flush.
        assert!(ack_rx.try_recv().is_err(), "no ack before flush");
        flush(&mut tracked).await;
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Ok(())));

        // Prevent Drop from racing: body already registered; IO drop drains empty.
        drop(body);
        drop(tracked);
    }

    #[tokio::test]
    async fn tracked_io_drop_before_flush_fails_ack() {
        let (tracked, _client, acks) = loopback_pair().await;
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let mut body = AckBody::new(b"x".to_vec(), ack_tx, acks);

        let mut cx = noop_cx();
        assert!(matches!(
            Pin::new(&mut body).poll_frame(&mut cx),
            Poll::Ready(Some(Ok(_)))
        ));
        // Registered with tracker; drop IO without flush → Cancelled/Err.
        drop(body);
        drop(tracked);
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Err(())));
    }

    #[tokio::test]
    async fn ack_body_drop_before_registration_fails() {
        let acks = FlushAcks::new();
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let body = AckBody::new(b"x".to_vec(), ack_tx, acks);
        drop(body);
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Err(())));
    }
}
