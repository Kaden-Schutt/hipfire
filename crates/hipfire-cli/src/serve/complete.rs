// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Completion pipeline and OpenAI translation.
//!
//! Concern: think-channel routing, semantic-event folding, stream-contract
//! gating, retry latches, and OpenAI adaptation (tool-call lifting, message
//! normalisation, finishing). Groups the deterministic, heavily-tested
//! transformations that convert daemon events into OpenAI responses.

use anyhow::{anyhow, bail, Context, Result};
use hipfire_client::ClientError;
use hipfire_runtime::prompt_frame::ToolCall;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, sync::{Arc, Mutex, mpsc}, thread, time::{Duration, Instant}};
use crate::serve::{ServeShared, ServeMeta, Admission, AdmissionGuard};
use crate::serve::slots;
use crate::{Paths, config_string, config_u64, config_bool, request_string, request_f64, request_u64, insert_optional_f64, insert_optional_u64, apply_http_reasoning_request, unix_timestamp};
use crate::serve::http::{request_id};

#[derive(Debug)]
pub(crate) struct Completion {
    pub(crate) id: String,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) content: String,
    pub(crate) reasoning_content: String,
    pub(crate) preserve_thinking: bool,
    pub(crate) tool_calls: Vec<ToolCall>,
    pub(crate) done: serde_json::Value,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ThinkFragment {
    Content(String),
    Reasoning(String),
}

#[derive(Debug, Default)]
pub(crate) struct ThinkChannelRouter {
    in_think: bool,
    pending: String,
    strip_answer_newlines: bool,
    semantic_split: bool,
    semantic_pending: String,
    semantic_reasoning: Option<bool>,
}

impl ThinkChannelRouter {
    pub(crate) fn set_started_in_think(&mut self, started: bool) {
        self.in_think = started;
    }

    pub(crate) fn push(&mut self, text: &str) -> Vec<ThinkFragment> {
        if self.semantic_split {
            return self.push_semantic(text, false);
        }
        self.pending.push_str(text);
        self.drain(false)
    }

    pub(crate) fn push_semantic(&mut self, text: &str, reasoning: bool) -> Vec<ThinkFragment> {
        let mut out = if self.pending.is_empty() {
            Vec::new()
        } else {
            self.drain(true)
        };
        self.semantic_split = true;
        if self.semantic_reasoning != Some(reasoning) {
            out.extend(self.drain_semantic(true));
            self.semantic_reasoning = Some(reasoning);
        }
        self.semantic_pending.push_str(text);
        out.extend(self.drain_semantic(false));
        out
    }

    pub(crate) fn finish(&mut self) -> Vec<ThinkFragment> {
        let mut out = self.drain(true);
        out.extend(self.drain_semantic(true));
        out
    }

    pub(crate) fn drain(&mut self, flush: bool) -> Vec<ThinkFragment> {
        pub(crate) const OPEN: &str = "<think>";
        pub(crate) const CLOSE: &str = "</think>";
        let mut out = Vec::new();
        loop {
            if let Some((index, marker)) = next_control_marker(&self.pending) {
                let before = self.pending[..index].to_owned();
                self.emit(before, &mut out);
                self.pending.drain(..index + marker.len());
                match marker {
                    OPEN => self.in_think = true,
                    CLOSE => {
                        self.in_think = false;
                        self.strip_answer_newlines = true;
                    }
                    _ => {}
                }
                continue;
            }

            let held = if flush {
                0
            } else {
                longest_control_prefix_suffix(&self.pending)
            };
            let emit_len = self.pending.len().saturating_sub(held);
            if emit_len > 0 {
                let text = self.pending[..emit_len].to_owned();
                self.pending.drain(..emit_len);
                self.emit(text, &mut out);
            }
            break;
        }
        out
    }

    pub(crate) fn drain_semantic(&mut self, flush: bool) -> Vec<ThinkFragment> {
        let mut out = Vec::new();
        loop {
            if let Some((index, marker)) = next_control_marker(&self.semantic_pending) {
                let before = self.semantic_pending[..index].to_owned();
                self.emit_semantic(before, &mut out);
                self.semantic_pending.drain(..index + marker.len());
                continue;
            }
            let held = if flush {
                0
            } else {
                longest_control_prefix_suffix(&self.semantic_pending)
            };
            let emit_len = self.semantic_pending.len().saturating_sub(held);
            if emit_len > 0 {
                let text = self.semantic_pending[..emit_len].to_owned();
                self.semantic_pending.drain(..emit_len);
                self.emit_semantic(text, &mut out);
            }
            break;
        }
        out
    }

    pub(crate) fn emit(&mut self, mut text: String, out: &mut Vec<ThinkFragment>) {
        if !self.in_think && self.strip_answer_newlines {
            let trimmed = text.trim_start_matches(['\r', '\n']);
            if trimmed.is_empty() {
                return;
            }
            text = trimmed.to_owned();
            self.strip_answer_newlines = false;
        }
        if text.is_empty() {
            return;
        }
        if self.in_think {
            out.push(ThinkFragment::Reasoning(text));
        } else {
            out.push(ThinkFragment::Content(text));
        }
    }

    pub(crate) fn emit_semantic(&self, text: String, out: &mut Vec<ThinkFragment>) {
        if text.is_empty() {
            return;
        }
        if self.semantic_reasoning == Some(true) {
            out.push(ThinkFragment::Reasoning(text));
        } else {
            out.push(ThinkFragment::Content(text));
        }
    }
}

pub(crate) const OUTPUT_CONTROL_MARKERS: &[&str] = &[
    "<think>",
    "</think>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|eot_id|>",
];

pub(crate) fn next_control_marker(text: &str) -> Option<(usize, &'static str)> {
    OUTPUT_CONTROL_MARKERS
        .iter()
        .filter_map(|marker| text.find(marker).map(|index| (index, *marker)))
        .min_by_key(|(index, _)| *index)
}

pub(crate) fn longest_control_prefix_suffix(text: &str) -> usize {
    OUTPUT_CONTROL_MARKERS
        .iter()
        .map(|marker| {
            let max = text.len().min(marker.len().saturating_sub(1));
            (1..=max)
                .rev()
                .find(|&len| text.ends_with(&marker[..len]))
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

/// Convert a daemon v2 structured tool-call JSON object into canonical [`ToolCall`].
pub(crate) fn tool_call_from_canonical_value(value: &serde_json::Value) -> Result<ToolCall, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "tool call must be a JSON object".to_owned())?;
    let name = obj
        .get("name")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "tool call missing non-empty name".to_owned())?
        .to_owned();
    let arguments = obj
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    Ok(ToolCall {
        id: None,
        name,
        arguments,
        rendered_body: None,
    })
}

/// Convert a retained legacy completion-boundary tool-call JSON value into
/// canonical [`ToolCall`] without marker parsing.
pub(crate) fn tool_call_from_legacy_value(value: &serde_json::Value) -> Result<ToolCall, String> {
    // Legacy wire already used `{name, arguments}` objects (same shape as v2).
    // Keep an explicit boundary so legacy retention never reintroduces text scans.
    tool_call_from_canonical_value(value)
}

/// Endpoint adapter kinds known to the serve HTTP surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EndpointAdapterKind {
    OpenAiChatCompletions,
}

/// Capability status of an endpoint adapter for non-empty tools requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EndpointAdapterStatus {
    /// Adapter is present and preserves canonical tool-call semantics losslessly.
    AvailableLossless,
    /// No adapter is registered for this endpoint.
    Unavailable,
    /// Adapter exists but would drop or rewrite tool-call semantics.
    Lossy,
}

/// Pre-generation denial when tools are requested without a safe adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EndpointAdapterError {
    Unavailable { endpoint: &'static str },
    Lossy { endpoint: &'static str },
}

impl std::fmt::Display for EndpointAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable { endpoint } => {
                write!(f, "endpoint adapter unavailable for tools on {endpoint}")
            }
            Self::Lossy { endpoint } => {
                write!(f, "endpoint adapter lossy for tools on {endpoint}")
            }
        }
    }
}

impl std::error::Error for EndpointAdapterError {}

/// Typed registry of HTTP endpoint adapters and their tool-call capability.
pub(crate) struct EndpointAdapterRegistry;

impl EndpointAdapterRegistry {
    pub(crate) fn status(kind: EndpointAdapterKind) -> EndpointAdapterStatus {
        match kind {
            // OpenAI chat completions lowering is present and lossless for ToolCall.
            EndpointAdapterKind::OpenAiChatCompletions => EndpointAdapterStatus::AvailableLossless,
        }
    }
}

pub(crate) fn endpoint_adapter_status(kind: EndpointAdapterKind) -> EndpointAdapterStatus {
    EndpointAdapterRegistry::status(kind)
}

/// Gate `/v1/chat/completions` when the request carries a non-empty `tools` array.
/// Tool-free requests are unchanged. Adapter availability never overrides producer safety.
pub(crate) fn gate_chat_completions_tools(body: &serde_json::Value) -> Result<(), EndpointAdapterError> {
    let has_tools = body
        .get("tools")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    if !has_tools {
        return Ok(());
    }
    match endpoint_adapter_status(EndpointAdapterKind::OpenAiChatCompletions) {
        EndpointAdapterStatus::AvailableLossless => Ok(()),
        EndpointAdapterStatus::Unavailable => Err(EndpointAdapterError::Unavailable {
            endpoint: "/v1/chat/completions",
        }),
        EndpointAdapterStatus::Lossy => Err(EndpointAdapterError::Lossy {
            endpoint: "/v1/chat/completions",
        }),
    }
}

/// Errors from request+attempt correlated semantic event folding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SemanticFoldError {
    /// Fold was used before `begin_attempt` established required ids.
    NoActiveAttempt,
    /// Event carried a different attempt id than the fold's active attempt.
    StaleAttempt { current: u64, got: u64 },
    /// Active attempt requires attempt_id on every subsequent event.
    MissingAttemptId { current: u64 },
    /// attempt_id was present but not a JSON number (u64 / non-neg i64).
    MalformedAttemptId { current: u64 },
    /// Event carried a different request id than the fold's active request.
    StaleRequestId { current: String, got: String },
    /// Active request requires nonempty string `id` on every subsequent event.
    MissingRequestId { current: String },
    /// `id` was present but empty or not a string.
    MalformedRequestId { current: String },
    /// Canonical tool-call payload failed structured conversion.
    MalformedToolCall { detail: String },
}

impl std::fmt::Display for SemanticFoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoActiveAttempt => {
                write!(f, "semantic fold requires begin_attempt before events")
            }
            Self::StaleAttempt { current, got } => {
                write!(f, "stale attempt event: current={current} got={got}")
            }
            Self::MissingAttemptId { current } => {
                write!(f, "missing attempt_id on event for attempt {current}")
            }
            Self::MalformedAttemptId { current } => {
                write!(f, "malformed attempt_id on event for attempt {current}")
            }
            Self::StaleRequestId { current, got } => {
                write!(f, "stale request id: current={current} got={got}")
            }
            Self::MissingRequestId { current } => {
                write!(f, "missing request id on event for request {current}")
            }
            Self::MalformedRequestId { current } => {
                write!(f, "malformed request id on event for request {current}")
            }
            Self::MalformedToolCall { detail } => {
                write!(f, "malformed canonical tool call: {detail}")
            }
        }
    }
}

impl std::error::Error for SemanticFoldError {}

/// Attempt-local pure fold over daemon **contract v2** semantic JSON events.
///
/// Accumulates clean content/reasoning verbatim (no marker scanning), buffers
/// structured tool calls until a tool-safe done, preserves the daemon
/// finish_reason, and rejects stale/missing/malformed attempt correlation from
/// the first event. Never invokes [`ThinkChannelRouter`].
///
/// Activate only when `gen_start.contract_version == 2`. Legacy (non-v2)
/// MiniMax/Cohere raw-think streams stay on the explicit ThinkChannelRouter
/// path outside this type.
#[derive(Debug, Default)]
pub(crate) struct SemanticEventFold {
    content: String,
    reasoning_content: String,
    buffered_tool_calls: Vec<ToolCall>,
    current_request_id: Option<String>,
    current_attempt_id: Option<u64>,
    done: Option<serde_json::Value>,
}

impl SemanticEventFold {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Start (or restart) a correlated request+attempt, clearing attempt-local state.
    /// Must be called with the allocated wire ids before the first event.
    pub(crate) fn begin_attempt(&mut self, request_id: impl Into<String>, attempt_id: u64) {
        self.current_request_id = Some(request_id.into());
        self.current_attempt_id = Some(attempt_id);
        self.content.clear();
        self.reasoning_content.clear();
        self.buffered_tool_calls.clear();
        self.done = None;
    }

    pub(crate) fn current_request_id(&self) -> Option<&str> {
        self.current_request_id.as_deref()
    }

    pub(crate) fn current_attempt_id(&self) -> Option<u64> {
        self.current_attempt_id
    }

    pub(crate) fn content(&self) -> &str {
        &self.content
    }

    pub(crate) fn reasoning_content(&self) -> &str {
        &self.reasoning_content
    }

    pub(crate) fn buffered_tool_calls(&self) -> &[ToolCall] {
        &self.buffered_tool_calls
    }

    pub(crate) fn done(&self) -> Option<&serde_json::Value> {
        self.done.as_ref()
    }

    /// Executable calls after a tool-safe terminal; empty otherwise.
    /// Prefer canonical `calls` embedded on the staged/final done payload
    /// (authoritative). Fall back to mid-stream buffered events only when the
    /// terminal omits `calls` (legacy producers).
    pub(crate) fn executable_tool_calls(&self) -> &[ToolCall] {
        match self
            .done
            .as_ref()
            .and_then(|done| done.get("finish_reason"))
            .and_then(serde_json::Value::as_str)
        {
            // Only the daemon's tool_calls terminal may release calls.
            Some("tool_calls") => &self.buffered_tool_calls,
            _ => &[],
        }
    }

    /// Parse canonical `calls` from a staged/final done envelope.
    /// When `finish_reason=tool_calls`, missing/non-array/malformed fails closed.
    /// Other finish reasons ignore `calls` (leave buffer untouched for withhold).
    pub(crate) fn absorb_terminal_calls(&mut self, done: &serde_json::Value) -> Result<(), SemanticFoldError> {
        let finish = done
            .get("finish_reason")
            .and_then(serde_json::Value::as_str);
        if finish != Some("tool_calls") {
            return Ok(());
        }
        let Some(calls_val) = done.get("calls") else {
            return Err(SemanticFoldError::MalformedToolCall {
                detail: "tool_calls terminal requires `calls` array on staged done".to_owned(),
            });
        };
        let Some(calls) = calls_val.as_array() else {
            return Err(SemanticFoldError::MalformedToolCall {
                detail: "tool_calls terminal `calls` must be a JSON array".to_owned(),
            });
        };
        // Authoritative staged payload — replace any previously buffered calls
        // so we never duplicate mid-stream + terminal arrays.
        let mut parsed = Vec::with_capacity(calls.len());
        for call in calls {
            let tc = tool_call_from_canonical_value(call)
                .map_err(|detail| SemanticFoldError::MalformedToolCall { detail })?;
            parsed.push(tc);
        }
        self.buffered_tool_calls = parsed;
        Ok(())
    }

    /// Parse attempt_id: JSON numbers only (u64 or non-neg i64). Distinguishes
    /// missing vs malformed (string / null / negative / object).
    pub(crate) fn parse_event_attempt_id(event: &serde_json::Value) -> Result<Option<u64>, SemanticFoldError> {
        match event.get("attempt_id") {
            None => Ok(None),
            Some(value) => {
                if let Some(n) = value.as_u64() {
                    return Ok(Some(n));
                }
                if let Some(n) = value.as_i64() {
                    if n >= 0 {
                        return Ok(Some(n as u64));
                    }
                }
                // Present but not a usable number — caller maps to Malformed.
                Err(SemanticFoldError::MalformedAttemptId {
                    current: 0, // placeholder; check_correlation overwrites with active id
                })
            }
        }
    }

    /// Parse request `id`: nonempty JSON string only. Distinguishes missing vs
    /// malformed (empty string / non-string).
    pub(crate) fn parse_event_request_id(
        event: &serde_json::Value,
    ) -> Result<Option<String>, SemanticFoldError> {
        match event.get("id") {
            None => Ok(None),
            Some(value) => match value.as_str() {
                Some(s) if !s.is_empty() => Ok(Some(s.to_owned())),
                Some(_) | None => Err(SemanticFoldError::MalformedRequestId {
                    current: String::new(),
                }),
            },
        }
    }

    pub(crate) fn check_correlation(&self, event: &serde_json::Value) -> Result<(), SemanticFoldError> {
        let current_attempt = self
            .current_attempt_id
            .ok_or(SemanticFoldError::NoActiveAttempt)?;
        let current_request = self
            .current_request_id
            .as_deref()
            .ok_or(SemanticFoldError::NoActiveAttempt)?;

        match Self::parse_event_request_id(event) {
            Ok(None) => {
                return Err(SemanticFoldError::MissingRequestId {
                    current: current_request.to_owned(),
                });
            }
            Ok(Some(got)) if got != current_request => {
                return Err(SemanticFoldError::StaleRequestId {
                    current: current_request.to_owned(),
                    got,
                });
            }
            Ok(Some(_)) => {}
            Err(SemanticFoldError::MalformedRequestId { .. }) => {
                return Err(SemanticFoldError::MalformedRequestId {
                    current: current_request.to_owned(),
                });
            }
            Err(other) => return Err(other),
        }

        match Self::parse_event_attempt_id(event) {
            Ok(None) => Err(SemanticFoldError::MissingAttemptId {
                current: current_attempt,
            }),
            Ok(Some(got)) if got != current_attempt => Err(SemanticFoldError::StaleAttempt {
                current: current_attempt,
                got,
            }),
            Ok(Some(_)) => Ok(()),
            Err(SemanticFoldError::MalformedAttemptId { .. }) => {
                Err(SemanticFoldError::MalformedAttemptId {
                    current: current_attempt,
                })
            }
            Err(other) => Err(other),
        }
    }

    /// Fold one daemon v2 semantic event. Returns logical events the caller may
    /// forward (token/reasoning fragments, done, …). Structured `tool_calls`
    /// are buffered and never returned for mid-stream forwarding.
    ///
    /// Token and reasoning text are appended **verbatim** — no marker scan.
    pub(crate) fn push(
        &mut self,
        event: &serde_json::Value,
    ) -> Result<Vec<serde_json::Value>, SemanticFoldError> {
        self.check_correlation(event)?;
        let mut forward = Vec::new();
        match event.get("type").and_then(serde_json::Value::as_str) {
            Some("gen_start") => {
                // v2 channels are typed; started_in_think is ignored (no marker router).
            }
            Some("token") => {
                if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                    // Daemon may tag reasoning on the token envelope; still verbatim.
                    if event.get("reasoning").and_then(serde_json::Value::as_bool) == Some(true) {
                        self.reasoning_content.push_str(text);
                        forward.push(serde_json::json!({ "type": "reasoning", "text": text }));
                    } else {
                        self.content.push_str(text);
                        forward.push(serde_json::json!({ "type": "token", "text": text }));
                    }
                }
            }
            Some("reasoning") => {
                if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                    self.reasoning_content.push_str(text);
                    forward.push(serde_json::json!({ "type": "reasoning", "text": text }));
                }
            }
            Some("tool_calls") => {
                let calls = event
                    .get("calls")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| SemanticFoldError::MalformedToolCall {
                        detail: "tool_calls event requires `calls` array".to_owned(),
                    })?;
                for call in calls {
                    let tc = tool_call_from_canonical_value(call)
                        .map_err(|detail| SemanticFoldError::MalformedToolCall { detail })?;
                    self.buffered_tool_calls.push(tc);
                }
                // Intentionally not forwarded — release only after tool-safe done.
            }
            Some("done") => {
                // Staged commit_ready is folded as type=done; absorb canonical
                // calls from the terminal payload before latching done.
                self.absorb_terminal_calls(event)?;
                self.done = Some(event.clone());
                forward.push(event.clone());
            }
            _ => {
                // Pass through unknown/control events (committed, error envelopes, …).
                forward.push(event.clone());
            }
        }
        Ok(forward)
    }
}

/// Allocate a fresh numeric generation attempt id (never 0 on success paths).
pub(crate) fn next_attempt_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Latched daemon event-contract for one `complete_request` stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamContract {
    /// Non-v2 / missing contract_version — ThinkChannelRouter path.
    Legacy,
    /// `gen_start.contract_version == 2` — SemanticEventFold path.
    V2,
}

/// Fail-closed stream framing / contract-selection errors for `complete_request`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StreamContractError {
    /// Stream opened with a non-`gen_start` event (no unchecked legacy default).
    PreStartEvent { event_type: String },
    /// More than one `gen_start` in a single attempt stream.
    SecondGenStart,
    /// `gen_start` lacked `attempt_id`.
    MissingAttemptId { expected: u64 },
    /// `gen_start.attempt_id` was present but not a usable number.
    MalformedAttemptId { expected: u64 },
    /// `gen_start.attempt_id` did not match the allocated wire id.
    StaleAttempt { expected: u64, got: u64 },
    /// `gen_start` lacked nonempty request `id`.
    MissingRequestId { expected: String },
    /// `gen_start.id` was present but empty or not a string.
    MalformedRequestId { expected: String },
    /// `gen_start.id` did not match the allocated wire request id.
    StaleRequestId { expected: String, got: String },
    /// Canonical tool-call payload failed structured conversion.
    MalformedToolCall { detail: String },
}

impl std::fmt::Display for StreamContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreStartEvent { event_type } => {
                write!(
                    f,
                    "stream must begin with gen_start; got {event_type} before contract latch"
                )
            }
            Self::SecondGenStart => {
                write!(f, "duplicate gen_start after contract already latched")
            }
            Self::MissingAttemptId { expected } => {
                write!(
                    f,
                    "gen_start missing attempt_id (expected {expected}); contract not latched"
                )
            }
            Self::MalformedAttemptId { expected } => {
                write!(
                    f,
                    "gen_start malformed attempt_id (expected {expected}); contract not latched"
                )
            }
            Self::StaleAttempt { expected, got } => {
                write!(
                    f,
                    "gen_start stale attempt_id: expected={expected} got={got}; contract not latched"
                )
            }
            Self::MissingRequestId { expected } => {
                write!(
                    f,
                    "gen_start missing request id (expected {expected}); contract not latched"
                )
            }
            Self::MalformedRequestId { expected } => {
                write!(
                    f,
                    "gen_start malformed request id (expected {expected}); contract not latched"
                )
            }
            Self::StaleRequestId { expected, got } => {
                write!(
                    f,
                    "gen_start stale request id: expected={expected} got={got}; contract not latched"
                )
            }
            Self::MalformedToolCall { detail } => {
                write!(f, "malformed canonical tool call: {detail}")
            }
        }
    }
}

impl std::error::Error for StreamContractError {}

/// One-shot contract latch for a generate stream.
///
/// Rules:
/// - first event must be exactly one `gen_start` with the expected numeric `attempt_id`
/// - correlation is validated **before** reading/latching `contract_version`
/// - legacy or v2 is latched once; a second `gen_start` is always rejected
/// - pre-start events never default to unchecked legacy
/// - after v2 is latched, nothing may switch the stream to legacy
#[derive(Debug)]
pub(crate) struct StreamContractGate {
    expected_request_id: String,
    expected_attempt_id: u64,
    latched: Option<StreamContract>,
}

impl StreamContractGate {
    pub(crate) fn new(expected_request_id: impl Into<String>, expected_attempt_id: u64) -> Self {
        Self {
            expected_request_id: expected_request_id.into(),
            expected_attempt_id,
            latched: None,
        }
    }

    pub(crate) fn contract(&self) -> Option<StreamContract> {
        self.latched
    }

    pub(crate) fn is_v2(&self) -> bool {
        self.latched == Some(StreamContract::V2)
    }

    /// Observe the next daemon event for framing/contract selection only.
    ///
    /// Returns the latched contract after a successful observe. Does not fold
    /// payloads — callers route to `SemanticEventFold` or legacy separately.
    pub(crate) fn observe(
        &mut self,
        event: &serde_json::Value,
    ) -> Result<StreamContract, StreamContractError> {
        let event_type = event
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<missing>");

        if let Some(contract) = self.latched {
            if event_type == "gen_start" {
                // Never re-latch / never allow a stale or missing-id start to
                // downgrade v2 → legacy (or flip legacy → v2).
                return Err(StreamContractError::SecondGenStart);
            }
            return Ok(contract);
        }

        // Unlatched: first event must be gen_start. No pre-start legacy default.
        if event_type != "gen_start" {
            return Err(StreamContractError::PreStartEvent {
                event_type: event_type.to_owned(),
            });
        }

        // Correlation BEFORE contract_version read/latch (exact id + attempt).
        let expected_request = self.expected_request_id.as_str();
        match SemanticEventFold::parse_event_request_id(event) {
            Ok(None) => {
                return Err(StreamContractError::MissingRequestId {
                    expected: expected_request.to_owned(),
                });
            }
            Ok(Some(got)) if got != expected_request => {
                return Err(StreamContractError::StaleRequestId {
                    expected: expected_request.to_owned(),
                    got,
                });
            }
            Ok(Some(_)) => {}
            Err(SemanticFoldError::MalformedRequestId { .. }) => {
                return Err(StreamContractError::MalformedRequestId {
                    expected: expected_request.to_owned(),
                });
            }
            Err(_) => {
                return Err(StreamContractError::MalformedRequestId {
                    expected: expected_request.to_owned(),
                });
            }
        }

        let expected = self.expected_attempt_id;
        match SemanticEventFold::parse_event_attempt_id(event) {
            Ok(None) => {
                return Err(StreamContractError::MissingAttemptId { expected });
            }
            Ok(Some(got)) if got != expected => {
                return Err(StreamContractError::StaleAttempt { expected, got });
            }
            Ok(Some(_)) => {}
            Err(SemanticFoldError::MalformedAttemptId { .. }) => {
                return Err(StreamContractError::MalformedAttemptId { expected });
            }
            Err(_) => {
                return Err(StreamContractError::MalformedAttemptId { expected });
            }
        }

        let contract = if event
            .get("contract_version")
            .and_then(serde_json::Value::as_u64)
            == Some(2)
        {
            StreamContract::V2
        } else {
            StreamContract::Legacy
        };
        self.latched = Some(contract);
        Ok(contract)
    }
}

/// Retry-disabling observations for one generation attempt.
///
/// Every event about to hit the client callback passes through [`Self::observe`]
/// (v2 fold logicals and legacy router fragments alike), so latching is
/// route-agnostic. `visible` matches exactly the wire-visible delta set of
/// [`openai_stream_delta_for_event`] (token/reasoning).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttemptLatches {
    visible: bool,
    commit_ready_seen: bool,
}

impl AttemptLatches {
    pub(crate) fn observe(&mut self, event: &serde_json::Value) {
        match event.get("type").and_then(serde_json::Value::as_str) {
            Some("token") | Some("reasoning") => self.visible = true,
            Some("commit_ready") => self.commit_ready_seen = true,
            _ => {}
        }
    }
}

/// Whether the failed attempt may be retried once server-side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetryDecision {
    Retry,
    Fail,
}

/// Single enforced retry-eligibility decision for the serve retry driver.
///
/// Retry iff ALL hold: gate enabled; first attempt; no visible token/reasoning
/// observed; commit_ready handshake never entered; daemon attested retry-reset
/// eligibility; the error is a typed daemon error of class `transient` with
/// `retryable=true` and an `attempt_id` matching the failed attempt. Every
/// other failure — malformed/validation/context/schema/tool/cancel classes,
/// callback/cancellation, I/O, EOF, invalid JSON, protocol/framing errors,
/// untyped legacy errors — fails closed with no retry.
pub(crate) fn decide_retry(
    error: &anyhow::Error,
    attempt_id: u64,
    latches: &AttemptLatches,
    eligible: bool,
    enabled: bool,
    attempt_index: u32,
) -> RetryDecision {
    if !enabled || attempt_index != 1 || latches.visible || latches.commit_ready_seen || !eligible {
        return RetryDecision::Fail;
    }
    let Some(hipfire_client::ClientError::Daemon(typed)) =
        error.downcast_ref::<hipfire_client::ClientError>()
    else {
        return RetryDecision::Fail;
    };
    if typed.class != hipfire_client::error_class::TRANSIENT
        || !typed.retryable
        || typed.attempt_id != attempt_id
    {
        return RetryDecision::Fail;
    }
    RetryDecision::Retry
}

/// Pure dual-route fold over a generate event sequence (test + `complete_request` core).
///
/// Applies [`StreamContractGate`] framing, then either [`SemanticEventFold`] (v2)
/// or legacy ThinkChannelRouter accumulation. Used by focused stream-contract
/// tests so production framing rules are exercised without a live Engine.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct FoldedStream {
    contract: StreamContract,
    content: String,
    reasoning_content: String,
    tool_calls: Vec<ToolCall>,
    done: Option<serde_json::Value>,
    #[allow(dead_code)]
    forwarded: Vec<serde_json::Value>,
}

#[cfg(test)]
pub(crate) fn fold_complete_request_stream(
    expected_request_id: &str,
    expected_attempt_id: u64,
    events: &[serde_json::Value],
) -> Result<FoldedStream, StreamContractError> {
    let mut fold = SemanticEventFold::new();
    fold.begin_attempt(expected_request_id, expected_attempt_id);
    let mut gate = StreamContractGate::new(expected_request_id, expected_attempt_id);
    let mut legacy_router = ThinkChannelRouter::default();
    let mut legacy_content = String::new();
    let mut legacy_reasoning = String::new();
    let mut legacy_tool_calls: Vec<ToolCall> = Vec::new();
    let mut legacy_done: Option<serde_json::Value> = None;
    let mut forwarded = Vec::new();

    for event in events {
        let contract = gate.observe(event)?;
        match contract {
            StreamContract::V2 => {
                // Map fold correlation errors onto stream framing errors for the
                // shared test surface (gate already validated gen_start).
                let logicals = fold.push(event).map_err(|err| match err {
                    SemanticFoldError::MissingAttemptId { current } => {
                        StreamContractError::MissingAttemptId { expected: current }
                    }
                    SemanticFoldError::MalformedAttemptId { current } => {
                        StreamContractError::MalformedAttemptId { expected: current }
                    }
                    SemanticFoldError::StaleAttempt { current, got } => {
                        StreamContractError::StaleAttempt {
                            expected: current,
                            got,
                        }
                    }
                    SemanticFoldError::MissingRequestId { current } => {
                        StreamContractError::MissingRequestId { expected: current }
                    }
                    SemanticFoldError::MalformedRequestId { current } => {
                        StreamContractError::MalformedRequestId { expected: current }
                    }
                    SemanticFoldError::StaleRequestId { current, got } => {
                        StreamContractError::StaleRequestId {
                            expected: current,
                            got,
                        }
                    }
                    SemanticFoldError::NoActiveAttempt => StreamContractError::PreStartEvent {
                        event_type: "no_active_attempt".into(),
                    },
                    SemanticFoldError::MalformedToolCall { detail } => {
                        StreamContractError::MalformedToolCall { detail }
                    }
                })?;
                for logical in logicals {
                    let ty = logical.get("type").and_then(serde_json::Value::as_str);
                    if ty == Some("done") || ty == Some("gen_start") {
                        continue;
                    }
                    forwarded.push(logical);
                }
            }
            StreamContract::Legacy => match event.get("type").and_then(serde_json::Value::as_str) {
                Some("gen_start") => {
                    if let Some(started) = event
                        .get("started_in_think")
                        .and_then(serde_json::Value::as_bool)
                    {
                        legacy_router.set_started_in_think(started);
                    }
                }
                Some("token") => {
                    if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                        let fragments =
                            if event.get("reasoning").and_then(serde_json::Value::as_bool)
                                == Some(true)
                            {
                                legacy_router.push_semantic(text, true)
                            } else {
                                legacy_router.push(text)
                            };
                        for fragment in fragments {
                            match fragment {
                                ThinkFragment::Content(t) => {
                                    legacy_content.push_str(&t);
                                    forwarded.push(serde_json::json!({
                                        "type": "token",
                                        "text": t
                                    }));
                                }
                                ThinkFragment::Reasoning(t) => {
                                    legacy_reasoning.push_str(&t);
                                    forwarded.push(serde_json::json!({
                                        "type": "reasoning",
                                        "text": t
                                    }));
                                }
                            }
                        }
                    }
                }
                Some("reasoning") => {
                    if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                        for fragment in legacy_router.push_semantic(text, true) {
                            match fragment {
                                ThinkFragment::Content(t) => {
                                    legacy_content.push_str(&t);
                                    forwarded.push(serde_json::json!({
                                        "type": "token",
                                        "text": t
                                    }));
                                }
                                ThinkFragment::Reasoning(t) => {
                                    legacy_reasoning.push_str(&t);
                                    forwarded.push(serde_json::json!({
                                        "type": "reasoning",
                                        "text": t
                                    }));
                                }
                            }
                        }
                    }
                }
                Some("tool_calls") => {
                    if let Some(calls) = event.get("calls").and_then(serde_json::Value::as_array) {
                        for call in calls {
                            let tc = tool_call_from_legacy_value(call).map_err(|detail| {
                                StreamContractError::MalformedToolCall { detail }
                            })?;
                            legacy_tool_calls.push(tc);
                        }
                    }
                }
                Some("done") => {
                    for fragment in legacy_router.finish() {
                        match fragment {
                            ThinkFragment::Content(t) => {
                                legacy_content.push_str(&t);
                                forwarded.push(serde_json::json!({
                                    "type": "token",
                                    "text": t
                                }));
                            }
                            ThinkFragment::Reasoning(t) => {
                                legacy_reasoning.push_str(&t);
                                forwarded.push(serde_json::json!({
                                    "type": "reasoning",
                                    "text": t
                                }));
                            }
                        }
                    }
                    legacy_done = Some(event.clone());
                }
                _ => {
                    forwarded.push(event.clone());
                }
            },
        }
    }

    let contract = gate
        .contract()
        .ok_or_else(|| StreamContractError::PreStartEvent {
            event_type: "<empty stream>".into(),
        })?;

    match contract {
        StreamContract::V2 => {
            let finish = fold
                .done()
                .and_then(|d| d.get("finish_reason"))
                .and_then(serde_json::Value::as_str);
            let tool_calls = if finish == Some("tool_calls") {
                fold.executable_tool_calls().to_vec()
            } else {
                Vec::new()
            };
            Ok(FoldedStream {
                contract,
                content: fold.content().to_owned(),
                reasoning_content: fold.reasoning_content().to_owned(),
                tool_calls,
                done: fold.done().cloned(),
                forwarded,
            })
        }
        StreamContract::Legacy => {
            let finish = legacy_done
                .as_ref()
                .and_then(|d| d.get("finish_reason"))
                .and_then(serde_json::Value::as_str);
            let tool_calls = if finish == Some("tool_calls") {
                legacy_tool_calls
            } else {
                Vec::new()
            };
            Ok(FoldedStream {
                contract,
                content: legacy_content,
                reasoning_content: legacy_reasoning,
                tool_calls,
                done: legacy_done,
                forwarded,
            })
        }
    }
}

/// One correlated generation attempt under the shared serve runtime lock.
///
/// `identity` is the public completion identity (stable across retries);
/// `attempt_id` is the freshly allocated wire attempt id for this attempt.
/// `force_reset` (retry attempts) cold-resets before generate; a failed forced
/// reset poisons cached model state so the next request full-reloads.
/// `latches` records retry-disabling observations for the driver.
pub(crate) fn complete_request_attempt(
    shared: &ServeShared,
    body: &serde_json::Value,
    guard: AdmissionGuard,
    identity: &(String, u64),
    attempt_id: u64,
    force_reset: bool,
    latches: &std::cell::RefCell<AttemptLatches>,
    event_callback: &mut dyn FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
    terminal_callback: &mut dyn FnMut(&Completion) -> Result<(), hipfire_client::ClientError>,
) -> Result<Completion> {
    // Gateway-side timing. The daemon emits neither ttft_ms nor latency_ms --
    // its `done` is {type,id,tokens,tok_s} -- so the gateway measures what it can
    // actually observe. Measuring here is arguably more honest anyway: this
    // interval includes admission queueing, which is exactly the component a
    // serving operator needs when latency rises under load.
    let attempt_started = std::time::Instant::now();
    let first_token_at: std::cell::Cell<Option<std::time::Duration>> =
        std::cell::Cell::new(None);

    // Latch retry-disabling observations on every event bound for the client.
    let mut event_callback = |event: &serde_json::Value| {
        if first_token_at.get().is_none()
            && event.get("type").and_then(serde_json::Value::as_str) == Some("token")
        {
            first_token_at.set(Some(attempt_started.elapsed()));
        }
        latches.borrow_mut().observe(event);
        event_callback(event)
    };
    let model = body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("model is required"))?
        .to_owned();
    let image_base64 = request_image_base64(body.get("messages"))?;
    // Acquire runtime, ensure model, and build the generate request while
    // holding the lock. Clone the engine handle before dropping the lock so
    // concurrent eligible requests can share the multiplexed transport.
    let (generate, resolved, engine_clone) = {
        let mut runtime = shared
            .runtime
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Attempt id is allocated by the retry driver before any cold reset /
        // generate so reset ack, generate request, and the semantic fold share one
        // wire id.
        let resolved = runtime.ensure_model(&model, &shared.meta, None)?;
        if force_reset || (!runtime.cache_capable && !runtime.continuous_batch_capable) {
            if let Err(error) = runtime.engine.reset(attempt_id) {
                if force_reset {
                    // Rollback could not be attested: model state is unknown, so
                    // the next request must full-reload rather than trust it.
                    runtime.current_path = None;
                    runtime.current_arch = None;
                    runtime.continuous_batch_capable = false;
                    runtime.current_max_seq = 0;
                    runtime.cache_capable = false;
                }
                return Err(error.into());
            }
        }
        let max_tokens = body
            .get("max_tokens")
            .or_else(|| body.get("max_completion_tokens"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(config_u64(&resolved, "generation.max_tokens")?);
        if max_tokens == 0 || max_tokens > 393_216 {
            bail!("max_tokens must be between 1 and 393216");
        }
        let required_max_seq = max_tokens.saturating_add(1024);
        if runtime.current_max_seq < required_max_seq {
            runtime.ensure_model(&model, &shared.meta, Some(required_max_seq))?;
        }
        let include_reasoning_content = runtime.current_arch.as_deref() == Some("muse_glimmer");
        let mut normalized_messages =
            normalize_openai_messages(body.get("messages"), include_reasoning_content);
        let default_system = request_string(&resolved, "prompt.system", None)?;
        inject_default_system_message(&mut normalized_messages, default_system.as_deref());
        let mut generate = serde_json::json!({
            "type": "generate",
            "id": request_id(),
            "prompt": last_user_prompt(&normalized_messages).unwrap_or_else(|| "Hello".into()),
            "messages": normalized_messages,
            "max_tokens": max_tokens,
            "attempt_id": attempt_id,
        });
        if let Some(image) = image_base64 {
            generate["image_base64"] = serde_json::Value::String(image);
        }
        for (key, config_key) in [
            ("temperature", "generation.temperature"),
            ("top_p", "generation.top_p"),
            ("repeat_penalty", "generation.repeat_penalty"),
        ] {
            let explicit = body.get(key).and_then(serde_json::Value::as_f64);
            insert_optional_f64(
                &mut generate,
                key,
                request_f64(&resolved, config_key, explicit)?,
            );
        }
        for name in [
            "tools",
            "tool_choice",
            "frequency_penalty",
            "stop",
            "reasoning_effort",
        ] {
            if let Some(value) = body.get(name) {
                generate[name] = value.clone();
            }
        }
        if let Some(value) = body.get("top_k") {
            generate["top_k"] = value.clone();
        } else {
            insert_optional_u64(
                &mut generate,
                "top_k",
                request_u64(&resolved, "generation.top_k", None)?,
            );
        }
        for (key, config_key) in [
            ("min_p", "generation.min_p"),
            ("presence_penalty", "generation.presence_penalty"),
        ] {
            if let Some(value) = body.get(key) {
                generate[key] = value.clone();
            } else {
                insert_optional_f64(
                    &mut generate,
                    key,
                    request_f64(&resolved, config_key, None)?,
                );
            }
        }
        let deepseek4_effort_contract = runtime.current_arch.as_deref() == Some("deepseek4");
        apply_http_reasoning_request(body, &resolved, &mut generate, deepseek4_effort_contract)?;
        let (id, created) = identity.clone();
        generate["id"] = serde_json::Value::String(id.clone());
        generate["attempt_id"] = serde_json::json!(attempt_id);
        if guard.is_eligible {
            generate["serve_continuous_batch"] = serde_json::Value::Bool(true);
        }
        let engine_clone = runtime.engine.clone();
        (generate, resolved, engine_clone)
    };
    let (id, created) = identity.clone();
    // Dual route gated by StreamContractGate:
    // - first event must be gen_start with matching request id + attempt_id
    // - contract_version is read only after correlation succeeds
    // - legacy/v2 latched once; second gen_start and pre-start events rejected
    // - v2 cannot be downgraded by a later stale/missing-id gen_start
    // - commit_ready is staged done: folded once via terminal_callback before commit
    let mut fold = SemanticEventFold::new();
    fold.begin_attempt(&id, attempt_id);
    let mut contract_gate = StreamContractGate::new(id.clone(), attempt_id);
    let mut legacy_router = ThinkChannelRouter::default();
    let mut legacy_content = String::new();
    let mut legacy_reasoning = String::new();
    let mut legacy_tool_calls: Vec<ToolCall> = Vec::new();
    let mut legacy_done: Option<serde_json::Value> = None;
    let mut terminal_delivered = false;
    let preserve_thinking = body
        .pointer("/chat_template_kwargs/preserve_thinking")
        .and_then(serde_json::Value::as_bool)
        == Some(true);
    let model_for_fold = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_owned();

    let gen_result = engine_clone.generate(&generate, |event| {
        let event_type = event.get("type").and_then(serde_json::Value::as_str);

        // Staged terminal: commit_ready carries done fields with type != done.
        // Fold/validate a type=done clone and deliver HTTP terminal before Ok.
        if event_type == Some("commit_ready") {
            latches.borrow_mut().commit_ready_seen = true;
            if terminal_delivered {
                return Err(hipfire_client::ClientError::Protocol(
                    "duplicate commit_ready".into(),
                ));
            }
            // Gate correlation on the raw envelope first (same id+attempt rules).
            let contract = contract_gate
                .observe(event)
                .map_err(|error| hipfire_client::ClientError::Protocol(error.to_string()))?;

            let mut staged = event.clone();
            if let Some(obj) = staged.as_object_mut() {
                obj.insert("type".into(), serde_json::Value::String("done".into()));
            } else {
                return Err(hipfire_client::ClientError::Protocol(
                    "commit_ready must be a JSON object".into(),
                ));
            }

            let preview = match contract {
                StreamContract::V2 => {
                    let forward = fold.push(&staged).map_err(|error| {
                        hipfire_client::ClientError::Protocol(error.to_string())
                    })?;
                    // Staged done is held on the fold; do not forward mid-stream.
                    let _ = forward;
                    Completion {
                        id: id.clone(),
                        created,
                        model: model.clone(),
                        content: fold.content().to_owned(),
                        reasoning_content: fold.reasoning_content().to_owned(),
                        preserve_thinking,
                        tool_calls: fold.executable_tool_calls().to_vec(),
                        done: fold.done().cloned().unwrap_or(staged),
                    }
                }
                StreamContract::Legacy => {
                    forward_think_fragments(
                        legacy_router.finish(),
                        &mut legacy_content,
                        &mut legacy_reasoning,
                        &mut event_callback,
                    )?;
                    legacy_done = Some(staged.clone());
                    let finish = staged
                        .get("finish_reason")
                        .and_then(serde_json::Value::as_str);
                    let tool_calls = if finish == Some("tool_calls") {
                        legacy_tool_calls.clone()
                    } else {
                        Vec::new()
                    };
                    Completion {
                        id: id.clone(),
                        created,
                        model: model.clone(),
                        content: legacy_content.clone(),
                        reasoning_content: legacy_reasoning.clone(),
                        preserve_thinking,
                        tool_calls,
                        done: staged,
                    }
                }
            };

            terminal_callback(&preview)?;
            terminal_delivered = true;
            return Ok(());
        }

        let contract = contract_gate
            .observe(event)
            .map_err(|error| hipfire_client::ClientError::Protocol(error.to_string()))?;

        match contract {
            StreamContract::V2 => {
                let forward = fold
                    .push(event)
                    .map_err(|error| hipfire_client::ClientError::Protocol(error.to_string()))?;
                for logical in forward {
                    // gen_start is consumed for latching; done is held on the fold.
                    // Post-commit done is not callback-visible from Engine, but
                    // still ignore if seen.
                    let ty = logical.get("type").and_then(serde_json::Value::as_str);
                    if ty == Some("done") || ty == Some("gen_start") {
                        continue;
                    }
                    event_callback(&logical)?;
                }
            }
            StreamContract::Legacy => {
                match event_type {
                    Some("gen_start") => {
                        if let Some(started) = event
                            .get("started_in_think")
                            .and_then(serde_json::Value::as_bool)
                        {
                            legacy_router.set_started_in_think(started);
                        }
                    }
                    Some("token") => {
                        if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                            let fragments =
                                if event.get("reasoning").and_then(serde_json::Value::as_bool)
                                    == Some(true)
                                {
                                    legacy_router.push_semantic(text, true)
                                } else {
                                    legacy_router.push(text)
                                };
                            forward_think_fragments(
                                fragments,
                                &mut legacy_content,
                                &mut legacy_reasoning,
                                &mut event_callback,
                            )?;
                        }
                    }
                    Some("reasoning") => {
                        if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                            let fragments = legacy_router.push_semantic(text, true);
                            forward_think_fragments(
                                fragments,
                                &mut legacy_content,
                                &mut legacy_reasoning,
                                &mut event_callback,
                            )?;
                        }
                    }
                    Some("tool_calls") => {
                        if let Some(calls) =
                            event.get("calls").and_then(serde_json::Value::as_array)
                        {
                            for call in calls {
                                let tc = tool_call_from_legacy_value(call).map_err(|detail| {
                                    hipfire_client::ClientError::Protocol(format!(
                                        "malformed canonical tool call: {detail}"
                                    ))
                                })?;
                                legacy_tool_calls.push(tc);
                            }
                        }
                    }
                    Some("done") => {
                        // Prefer staged commit_ready terminal; keep post-commit done
                        // only as payload fill if staging was skipped (legacy path).
                        if !terminal_delivered {
                            forward_think_fragments(
                                legacy_router.finish(),
                                &mut legacy_content,
                                &mut legacy_reasoning,
                                &mut event_callback,
                            )?;
                            legacy_done = Some(event.clone());
                        }
                    }
                    _ => {
                        event_callback(event)?;
                    }
                }
            }
        }
        Ok(())
    });
    let done = gen_result?;
    let mut meta = shared
        .meta
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    meta.requests_served = meta.requests_served.saturating_add(1);
    meta.recent_tok_s = done.get("tok_s").and_then(serde_json::Value::as_f64);
    // Same payload the gateway already had; previously only `tok_s` survived, as a
    // scalar overwritten by the next request. Distributions are what make a stall
    // visible.
    shared.metrics.observe_done(&done);
    shared.metrics.observe_timing(
        first_token_at.get().map(|d| d.as_secs_f64() * 1000.0),
        attempt_started.elapsed().as_secs_f64() * 1000.0,
    );
    meta.last_activity = Instant::now();

    if contract_gate.is_v2() {
        let done = fold.done().cloned().unwrap_or(done);
        return Ok(Completion {
            id,
            created,
            model,
            content: fold.content().to_owned(),
            reasoning_content: fold.reasoning_content().to_owned(),
            preserve_thinking,
            tool_calls: fold.executable_tool_calls().to_vec(),
            done,
        });
    }

    let done = legacy_done.unwrap_or(done);
    let finish = done
        .get("finish_reason")
        .and_then(serde_json::Value::as_str);
    let tool_calls = if finish == Some("tool_calls") {
        legacy_tool_calls
    } else {
        Vec::new()
    };
    Ok(Completion {
        id,
        created,
        model,
        content: legacy_content,
        reasoning_content: legacy_reasoning,
        preserve_thinking,
        tool_calls,
        done,
    })
}

/// Server-owned one-retry driver over [`complete_request_attempt`].

pub(crate) fn complete_request(
    shared: &ServeShared,
    body: &serde_json::Value,
    guard: AdmissionGuard,
    request_identity: Option<(String, u64)>,
    mut event_callback: impl FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
    mut terminal_callback: impl FnMut(&Completion) -> Result<(), hipfire_client::ClientError>,
) -> Result<Completion> {
    let identity = request_identity.unwrap_or_else(|| (request_id(), unix_timestamp()));

    // Concurrent path. Takes no `shared.runtime` lock, which is the entire
    // point: that mutex is what makes concurrent callers queue today. The
    // retry/attempt machinery below is for the daemon Engine's cold-reset
    // semantics and has no analogue here -- the engine either admits a request
    // or rejects it with a reason.
    if let Some(backend) = shared.slot_engine.clone() {
        return crate::serve::slots::complete_request_slots(
            &backend,
            body,
            &identity,
            &mut event_callback,
            &mut terminal_callback,
        );
    }

    let mut attempt_index = 1u32;
    let mut guard = guard;
    loop {
        let attempt_id = next_attempt_id();
        let latches = std::cell::RefCell::new(AttemptLatches::default());
        let outcome = complete_request_attempt(
            shared,
            body,
            guard,
            &identity,
            attempt_id,
            attempt_index > 1,
            &latches,
            &mut event_callback,
            &mut terminal_callback,
        );
        let latches = latches.into_inner();
        match outcome {
            Ok(completion) => {
                if attempt_index > 1 {
                    let mut meta = shared
                        .meta
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    meta.retries_succeeded = meta.retries_succeeded.saturating_add(1);
                }
                return Ok(completion);
            }
            Err(error) => {
                let eligible = {
                    let runtime = shared
                        .runtime
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    runtime.engine.last_retry_reset_eligible() == Some(true)
                };
                if decide_retry(
                    &error,
                    attempt_id,
                    &latches,
                    eligible,
                    shared.retry_enabled,
                    attempt_index,
                ) == RetryDecision::Fail
                {
                    return Err(error);
                }
                {
                    let mut meta = shared
                        .meta
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    meta.retries_attempted = meta.retries_attempted.saturating_add(1);
                }
                eprintln!(
                    "[hipfire] {}: typed transient daemon failure on attempt {attempt_index}; \
                     rolling back and retrying once",
                    identity.0
                );
                let hook = shared
                    .backoff_hook
                    .lock()
                    .ok()
                    .and_then(|guard| guard.clone());
                if let Some(hook) = hook {
                    hook(shared.retry_backoff);
                } else {
                    std::thread::sleep(shared.retry_backoff);
                }
                guard = match shared.admission.acquire() {
                    Ok(guard) => guard,
                    Err(_) => {
                        return Err(error.context("retry aborted: admission re-acquire failed"));
                    }
                };
                attempt_index = attempt_index.saturating_add(1);
            }
        }
    }
}

pub(crate) fn forward_think_fragments(
    fragments: Vec<ThinkFragment>,
    content: &mut String,
    reasoning_content: &mut String,
    event_callback: &mut impl FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
) -> Result<(), hipfire_client::ClientError> {
    for fragment in fragments {
        let logical = match fragment {
            ThinkFragment::Content(text) => {
                content.push_str(&text);
                serde_json::json!({ "type": "token", "text": text })
            }
            ThinkFragment::Reasoning(text) => {
                reasoning_content.push_str(&text);
                serde_json::json!({ "type": "reasoning", "text": text })
            }
        };
        event_callback(&logical)?;
    }
    Ok(())
}

pub(crate) fn openai_content_text(content: Option<&serde_json::Value>) -> String {
    match content {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter(|part| part.get("type").and_then(serde_json::Value::as_str) == Some("text"))
            .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
            .collect(),
        Some(other) => other.to_string(),
    }
}

pub(crate) fn request_image_base64(messages: Option<&serde_json::Value>) -> Result<Option<String>> {
    let Some(messages) = messages.and_then(serde_json::Value::as_array) else {
        return Ok(None);
    };
    let mut image = None;
    for message in messages {
        let Some(parts) = message.get("content").and_then(serde_json::Value::as_array) else {
            continue;
        };
        for part in parts {
            if part.get("type").and_then(serde_json::Value::as_str) != Some("image_url") {
                continue;
            }
            let url = part
                .pointer("/image_url/url")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow!("image_url content part requires image_url.url"))?;
            let payload = ["data:image/png;base64,", "data:image/jpeg;base64,"]
                .into_iter()
                .find_map(|prefix| url.strip_prefix(prefix))
                .ok_or_else(|| {
                    if url.starts_with("data:") {
                        anyhow!("only base64 PNG and JPEG image_url data URIs are supported")
                    } else {
                        anyhow!("remote image_url values are unsupported; send a base64 data URI")
                    }
                })?;
            if payload.is_empty() {
                bail!("image_url data URI has an empty base64 payload");
            }
            if image.replace(payload.to_owned()).is_some() {
                bail!("at most one image_url is supported per request");
            }
        }
    }
    Ok(image)
}

pub(crate) fn strip_inline_thinking(text: &str) -> String {
    pub(crate) const OPEN: &str = "<think>";
    pub(crate) const CLOSE: &str = "</think>";
    let mut visible = String::new();
    let mut remaining = text;
    while let Some(open) = remaining.find(OPEN) {
        visible.push_str(&remaining[..open]);
        let after_open = &remaining[open + OPEN.len()..];
        let Some(close) = after_open.find(CLOSE) else {
            return visible;
        };
        remaining = after_open[close + CLOSE.len()..].trim_start();
    }
    visible.push_str(remaining);
    visible
}

pub(crate) fn inline_thinking(text: &str) -> Option<String> {
    pub(crate) const OPEN: &str = "<think>";
    pub(crate) const CLOSE: &str = "</think>";
    let after_open = text.split_once(OPEN)?.1;
    let reasoning = after_open.split_once(CLOSE)?.0.trim();
    (!reasoning.is_empty()).then(|| reasoning.to_owned())
}

pub(crate) fn normalize_openai_tool_call(
    call: &serde_json::Value,
    include_reasoning_content: bool,
) -> serde_json::Value {
    let function = call.get("function").unwrap_or(call);
    let name = function
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let id = call
        .get("id")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty());
    let arguments = match function.get("arguments") {
        Some(serde_json::Value::String(raw)) => {
            match serde_json::from_str::<serde_json::Value>(raw) {
                // Muse Glimmer's Onyx template requires a MAPPING and calls `raise_exception`
                // otherwise, which would take the whole render down to a bare unframed prompt.
                // Surface a parsed-but-not-object payload as the raw string so the daemon's
                // `normalize_glimmer_tool_arguments` can refuse the request loudly instead.
                Ok(parsed) if include_reasoning_content && !parsed.is_object() => {
                    serde_json::Value::String(raw.clone())
                }
                // Every other architecture keeps whatever parsed — an array or scalar argument
                // payload is legal for them, and `_raw`-wrapping it here would invent a tool
                // parameter the model never saw. This arm is load-bearing for no-clobber.
                Ok(parsed) => parsed,
                Err(_) => {
                    if include_reasoning_content {
                        serde_json::Value::String(raw.clone())
                    } else {
                        serde_json::json!({ "_raw": raw })
                    }
                }
            }
        }
        Some(value) => value.clone(),
        None => serde_json::json!({}),
    };
    let mut obj = serde_json::json!({ "name": name, "arguments": arguments });
    if let Some(id_str) = id {
        obj["id"] = serde_json::Value::String(id_str.to_owned());
    }
    obj
}

pub(crate) fn normalize_openai_messages(
    messages: Option<&serde_json::Value>,
    include_reasoning_content: bool,
) -> serde_json::Value {
    let Some(messages) = messages.and_then(serde_json::Value::as_array) else {
        return serde_json::json!([]);
    };
    let normalized = messages
        .iter()
        .filter_map(|message| {
            let role = match message.get("role").and_then(serde_json::Value::as_str)? {
                "developer" => "system",
                "toolResult" | "tool_result" => "tool",
                role @ ("system" | "user" | "assistant" | "tool") => role,
                _ => return None,
            };
            let raw_content = openai_content_text(message.get("content"));
            let mut entry = serde_json::json!({
                "role": role,
                "content": if role == "assistant" {
                    strip_inline_thinking(&raw_content)
                } else {
                    raw_content.clone()
                },
            });
            if role == "assistant" {
                let reasoning = message
                    .get("reasoning")
                    .and_then(serde_json::Value::as_str)
                    .filter(|text| !text.is_empty())
                    .or_else(|| {
                        message
                            .get("reasoning_content")
                            .and_then(serde_json::Value::as_str)
                            .filter(|text| !text.is_empty())
                    })
                    .map(str::to_owned)
                    .or_else(|| inline_thinking(&raw_content));
                if let Some(reasoning) = reasoning {
                    if include_reasoning_content {
                        entry["reasoning_content"] = serde_json::Value::String(reasoning.clone());
                    }
                    entry["tool_plan"] = serde_json::Value::String(reasoning);
                }
                if let Some(calls) = message
                    .get("tool_calls")
                    .and_then(serde_json::Value::as_array)
                    .filter(|calls| !calls.is_empty())
                {
                    entry["tool_calls"] = serde_json::Value::Array(
                        calls
                            .iter()
                            .map(|c| normalize_openai_tool_call(c, include_reasoning_content))
                            .collect(),
                    );
                }
            } else if role == "tool" {
                if let Some(tool_call_id) = message
                    .get("tool_call_id")
                    .and_then(serde_json::Value::as_str)
                    .filter(|id| !id.is_empty())
                {
                    entry["tool_call_id"] = serde_json::Value::String(tool_call_id.to_owned());
                }
                if let Some(name) = message
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .filter(|n| !n.is_empty())
                {
                    entry["name"] = serde_json::Value::String(name.to_owned());
                }
            }
            Some(entry)
        })
        .collect();
    serde_json::Value::Array(normalized)
}

pub(crate) fn inject_default_system_message(messages: &mut serde_json::Value, system: Option<&str>) {
    let Some(system) = system.filter(|value| !value.is_empty()) else {
        return;
    };
    let Some(messages) = messages.as_array_mut() else {
        return;
    };
    if messages
        .iter()
        .any(|message| message.get("role").and_then(serde_json::Value::as_str) == Some("system"))
    {
        return;
    }
    messages.insert(
        0,
        serde_json::json!({ "role": "system", "content": system }),
    );
}

pub(crate) fn last_user_prompt(messages: &serde_json::Value) -> Option<String> {
    messages
        .as_array()?
        .iter()
        .rev()
        .find(|message| message.get("role").and_then(serde_json::Value::as_str) == Some("user"))
        .and_then(|message| message.get("content"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

pub(crate) fn openai_finish_reason(done: &serde_json::Value) -> String {
    // Only an explicit raw daemon finish_reason string is authoritative.
    // Never synthesize "tool_calls" from buffered/leaked calls when the
    // terminal is missing, null, non-string, or any other unsafe value.
    match done
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
    {
        Some(reason) => reason.to_owned(),
        // Missing/null/non-string → fail closed to stop (not tool_calls).
        None => "stop".into(),
    }
}

pub(crate) fn completion_json(completion: &Completion) -> serde_json::Value {
    let finish_reason = openai_finish_reason(&completion.done);
    // Structured calls only for a tool-safe terminal; never on length/error/cancel.
    let tool_calls = if finish_reason == "tool_calls" {
        openai_tool_calls(&completion.tool_calls)
    } else {
        Vec::new()
    };
    let visible_content =
        if completion.preserve_thinking && !completion.reasoning_content.is_empty() {
            format!(
                "<think>{}</think>\n{}",
                completion.reasoning_content, completion.content
            )
        } else {
            completion.content.clone()
        };
    // Pure tool turns: OpenAI content is JSON null (not "").
    let content_value = if visible_content.is_empty() && !tool_calls.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::String(visible_content)
    };
    let mut message = serde_json::json!({
        "role": "assistant",
        "content": content_value,
    });
    if !completion.preserve_thinking && !completion.reasoning_content.is_empty() {
        message["reasoning_content"] =
            serde_json::Value::String(completion.reasoning_content.clone());
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = serde_json::Value::Array(tool_calls);
    }
    serde_json::json!({
        "id": completion.id,
        "object": "chat.completion",
        "created": completion.created,
        "model": completion.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": completion_usage(completion),
        "timings": completion_timings(completion),
        "hipfire": completion_hipfire(completion),
    })
}

pub(crate) fn completion_usage(completion: &Completion) -> serde_json::Value {
    let cached_tokens = completion
        .done
        .get("cached_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let prompt_tokens = completion
        .done
        .get("prompt_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| {
            completion
                .done
                .get("prefill_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                .saturating_add(cached_tokens)
        });
    let completion_tokens = completion
        .done
        .get("tokens")
        .or_else(|| completion.done.get("completion_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    serde_json::json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": { "cached_tokens": cached_tokens },
    })
}

pub(crate) fn completion_timings(completion: &Completion) -> serde_json::Value {
    let done = &completion.done;
    serde_json::json!({
        "ttft_ms": done.get("ttft_ms"),
        "prefill_ms": done.get("prefill_ms"),
        "prefill_tok_s": done.get("prefill_tok_s"),
        "decode_tok_s": done.get("decode_tok_s").or_else(|| done.get("tok_s")),
        "latency_ms": done.get("latency_ms"),
        "tau": done.get("tau"),
        "cycles": done.get("cycles"),
        "dflash": done.get("dflash"),
        "mtp": done.get("mtp"),
        "mtp_ngram": done.get("mtp_ngram"),
        "ngram_mod_windows": done.get("ngram_mod_windows"),
        "ngram_mod_drafts": done.get("ngram_mod_drafts"),
        "ngram_mod_accepted": done.get("ngram_mod_accepted"),
        "ngram_mod_accept_rate": done.get("ngram_mod_accept_rate"),
        "mtp_windows": done.get("mtp_windows"),
        "ar_windows": done.get("ar_windows"),
        "mtp_retired": done.get("mtp_retired"),
        "mtp_window_timings": done.get("mtp_window_timings"),
    })
}

/// Shared hipfire evidence projection for normal and streaming terminals.
pub(crate) fn completion_hipfire(completion: &Completion) -> serde_json::Value {
    let done = &completion.done;
    serde_json::json!({
        "tok_s": done.get("tok_s"),
        "prefill_tok_s": done.get("prefill_tok_s"),
        "decode_tok_s": done.get("decode_tok_s"),
        "execution_mode": done.get("execution_mode"),
        "continuous_batch": done.get("continuous_batch"),
    })
}

/// One OpenAI-lowered tool call from the shared canonical adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OpenAiToolCallAdapterResult {
    pub(crate) index: usize,
    pub(crate) id: String,
    pub(crate) name: String,
    /// JSON-text arguments (OpenAI wire requires a string, not an object).
    pub arguments: String,
}

/// Build the single shared OpenAI adapter result vector for a completion.
/// Deterministic response-scoped ids `call_{index}`; no filtering/dropping.
pub(crate) fn openai_tool_call_adapter_results(calls: &[ToolCall]) -> Vec<OpenAiToolCallAdapterResult> {
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| OpenAiToolCallAdapterResult {
            index,
            id: format!("call_{index}"),
            name: call.name.clone(),
            arguments: serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into()),
        })
        .collect()
}

/// Lower shared adapter results into OpenAI `message.tool_calls` objects.
pub(crate) fn openai_tool_calls_from_adapter(
    adapted: &[OpenAiToolCallAdapterResult],
) -> Vec<serde_json::Value> {
    adapted
        .iter()
        .map(|call| {
            serde_json::json!({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                }
            })
        })
        .collect()
}

/// Canonical → OpenAI non-stream tool_calls array (shared adapter path).
pub(crate) fn openai_tool_calls(calls: &[ToolCall]) -> Vec<serde_json::Value> {
    openai_tool_calls_from_adapter(&openai_tool_call_adapter_results(calls))
}

/// Map a folded callback event to an OpenAI stream delta.
/// Only clean content/reasoning are forwarded mid-stream; structured tool
/// calls release only via [`openai_stream_terminal_chunks`].
pub(crate) fn openai_stream_delta_for_event(event: &serde_json::Value) -> Option<serde_json::Value> {
    match event.get("type").and_then(serde_json::Value::as_str) {
        Some("token") => event
            .get("text")
            .and_then(serde_json::Value::as_str)
            .map(|text| serde_json::json!({ "content": text })),
        Some("reasoning") => event
            .get("text")
            .and_then(serde_json::Value::as_str)
            .map(|text| serde_json::json!({ "reasoning_content": text })),
        // tool_calls are released only after a tool-safe terminal verdict.
        Some("tool_calls") => None,
        _ => None,
    }
}

/// Lower shared adapter results into an OpenAI stream `delta` tool_calls object.
pub(crate) fn openai_tool_call_delta_from_adapter(
    adapted: &[OpenAiToolCallAdapterResult],
) -> serde_json::Value {
    serde_json::json!({
        "tool_calls": adapted
            .iter()
            .map(|call| {
                serde_json::json!({
                    "index": call.index,
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                })
            })
            .collect::<Vec<_>>()
    })
}

/// Post-completion SSE chunks: optional tool_calls release, terminal choice,
/// then optional separate `choices: []` usage chunk (never on the terminal).
pub(crate) fn openai_stream_terminal_chunks(
    completion: &Completion,
    include_usage: bool,
) -> Vec<serde_json::Value> {
    let finish_reason = openai_finish_reason(&completion.done);
    let mut chunks = Vec::new();

    // Release structured calls only on a tool-safe terminal.
    // Same adapter vector as non-stream `openai_tool_calls`.
    if finish_reason == "tool_calls" && !completion.tool_calls.is_empty() {
        let adapted = openai_tool_call_adapter_results(&completion.tool_calls);
        let delta = openai_tool_call_delta_from_adapter(&adapted);
        chunks.push(serde_json::json!({
            "id": completion.id,
            "object": "chat.completion.chunk",
            "created": completion.created,
            "model": completion.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": null
            }],
        }));
    }

    chunks.push(serde_json::json!({
        "id": completion.id,
        "object": "chat.completion.chunk",
        "created": completion.created,
        "model": completion.model,
        "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason }],
        "timings": completion_timings(completion),
        "hipfire": completion_hipfire(completion),
    }));

    if include_usage {
        chunks.push(serde_json::json!({
            "id": completion.id,
            "object": "chat.completion.chunk",
            "created": completion.created,
            "model": completion.model,
            "choices": [],
            "usage": completion_usage(completion),
        }));
    }

    chunks
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::complete::{Completion};
    use hipfire_runtime::prompt_frame::ToolCall;
    use hipfire_client::ClientError;
    use crate::{Paths, ServeArgs, StopArgs};
    use hipfire_config::{ConfigLayer, ConfigPaths, ConfigSource, NamedLayer, resolve, load_global};
    use hipfire_registry::{RegistryV1, RegistryPaths};
    use std::{env, fs, path::{Path, PathBuf}, time::{Duration, Instant}, sync::{Arc, Mutex}};

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

    fn sample_completion(
        content: &str,
        tool_calls: Vec<ToolCall>,
        finish_reason: &str,
    ) -> Completion {
        Completion {
            id: "chatcmpl_test".into(),
            created: 42,
            model: "qwen:test".into(),
            content: content.into(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls,
            done: serde_json::json!({
                "finish_reason": finish_reason,
                "prompt_tokens": 3,
                "tokens": 5,
                "cached_tokens": 1,
                "tok_s": 10.0,
            }),
        }
    }

    fn sample_tc(name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: None,
            name: name.into(),
            arguments,
            rendered_body: None,
        }
    }

    fn sample_completion_with_done(
        content: &str,
        tool_calls: Vec<ToolCall>,
        done: serde_json::Value,
    ) -> Completion {
        Completion {
            id: "chatcmpl_test".into(),
            created: 42,
            model: "qwen:test".into(),
            content: content.into(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls,
            done,
        }
    }

    fn sample_tool_call(name: &str) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "arguments": { "path": "README.md" }
        })
    }

    fn task15_daemon_err(class: &str, retryable: bool, attempt_id: u64) -> anyhow::Error {
        anyhow::Error::new(hipfire_client::ClientError::Daemon(
            hipfire_client::TypedDaemonError {
                message: format!("t15 {class}"),
                class: class.to_owned(),
                retryable,
                rolled_back: false,
                attempt_id,
                id: Some("req-t15".into()),
            },
        ))
    }

    #[test]
    fn completion_timings_preserves_speculator_identity() {
        let completion = |done| Completion {
            id: "req-test".into(),
            created: 0,
            model: "test-model".into(),
            content: String::new(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done,
        };

        let dflash = completion_timings(&completion(serde_json::json!({
            "dflash": true,
            "tau": 3.5,
            "cycles": 4,
        })));
        assert_eq!(dflash["dflash"], true);
        assert!(dflash["mtp"].is_null());

        let mtp_window_timings = serde_json::json!([{
            "kind": "mtp",
            "wall_us": 1234,
            "draft_lookup_us": 12,
            "launch_us": 34,
            "h2d_us": 56,
            "d2h_us": 78,
            "d2d_us": 90,
            "memset_us": 11,
            "stream_sync_us": 22,
            "event_sync_us": 33,
            "device_sync_us": 44,
            "graph_launch_us": 55,
        }]);
        let mtp = completion_timings(&completion(serde_json::json!({
            "mtp": true,
            "tau": 2.0,
            "cycles": 6,
            "mtp_window_timings": mtp_window_timings,
        })));
        assert!(mtp["dflash"].is_null());
        assert_eq!(mtp["mtp"], true);
        assert_eq!(mtp["mtp_window_timings"], mtp_window_timings);
        assert_eq!(mtp["mtp_window_timings"][0]["kind"], "mtp");
        assert_eq!(mtp["mtp_window_timings"][0]["wall_us"], 1234);
    }

    #[test]
    fn completion_timings_projects_latency_ms() {
        let completion = Completion {
            id: "req-lat".into(),
            created: 0,
            model: "test-model".into(),
            content: String::new(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done: serde_json::json!({
                "ttft_ms": 12.5,
                "prefill_ms": 10.0,
                "prefill_tok_s": 100.0,
                "decode_tok_s": 40.0,
                "latency_ms": 250.5,
                "tok_s": 20.0,
            }),
        };
        let timings = completion_timings(&completion);
        assert_eq!(timings["latency_ms"], 250.5);
        assert_eq!(timings["ttft_ms"], 12.5);
        // Absent latency stays null-like (serde_json::Value::Null via .get).
        let no_lat = completion_timings(&Completion {
            id: "req-lat".into(),
            created: 0,
            model: "test-model".into(),
            content: String::new(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done: serde_json::json!({"ttft_ms": 1.0}),
        });
        assert!(no_lat["latency_ms"].is_null());
    }

    #[test]
    fn completion_hipfire_projects_batch_route_evidence() {
        let batch = serde_json::json!({
            "executed": true,
            "slots": 4,
            "lane": 1,
            "lane_capacity": 4096,
            "max_active_lanes": 2,
            "refill": "continuous",
        });
        let completion = Completion {
            id: "req-batch".into(),
            created: 0,
            model: "test-model".into(),
            content: "hi".into(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done: serde_json::json!({
                "tok_s": 18.5,
                "prefill_tok_s": 200.0,
                "decode_tok_s": 22.0,
                "latency_ms": 300.0,
                "execution_mode": "continuous_batch_independent",
                "continuous_batch": batch,
                "finish_reason": "stop",
                "tokens": 3,
                "prefill_tokens": 8,
            }),
        };
        let hip = completion_hipfire(&completion);
        assert_eq!(hip["execution_mode"], "continuous_batch_independent");
        assert_eq!(hip["continuous_batch"]["max_active_lanes"], 2);
        assert_eq!(hip["continuous_batch"]["slots"], 4);
        assert_eq!(hip["continuous_batch"]["lane"], 1);
        assert_eq!(hip["tok_s"], 18.5);

        let json = completion_json(&completion);
        assert_eq!(json["timings"]["latency_ms"], 300.0);
        assert_eq!(
            json["hipfire"]["execution_mode"],
            "continuous_batch_independent"
        );
        assert_eq!(json["hipfire"]["continuous_batch"]["refill"], "continuous");

        let chunks = openai_stream_terminal_chunks(&completion, false);
        let terminal = chunks.last().unwrap();
        assert_eq!(terminal["timings"]["latency_ms"], 300.0);
        assert_eq!(
            terminal["hipfire"]["execution_mode"],
            "continuous_batch_independent"
        );
        assert_eq!(
            terminal["hipfire"]["continuous_batch"]["max_active_lanes"],
            2
        );

        // Sequential done without route evidence stays null (unchanged fields).
        let sequential = Completion {
            id: "req-seq".into(),
            created: 0,
            model: "test-model".into(),
            content: String::new(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done: serde_json::json!({
                "tok_s": 10.0,
                "prefill_tok_s": 50.0,
                "decode_tok_s": 12.0,
                "finish_reason": "stop",
                "tokens": 2,
            }),
        };
        let seq_hip = completion_hipfire(&sequential);
        assert!(seq_hip["execution_mode"].is_null());
        assert!(seq_hip["continuous_batch"].is_null());
        assert_eq!(seq_hip["tok_s"], 10.0);
    }

    #[test]
    fn last_user_prompt_handles_text_parts() {
        let body = serde_json::json!({
            "messages": [
                { "role": "assistant", "content": "old" },
                { "role": "user", "content": [
                    { "type": "text", "text": "one" },
                    { "type": "text", "text": "two" }
                ] }
            ]
        });
        let messages = normalize_openai_messages(body.get("messages"), false);
        assert_eq!(last_user_prompt(&messages).as_deref(), Some("onetwo"));
    }

    #[test]
    fn openai_images_forward_one_base64_payload_and_reject_unsafe_shapes() {
        let messages = serde_json::json!([{
            "role": "user",
            "content": [
                { "type": "text", "text": "describe" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,YWJj" } }
            ]
        }]);
        assert_eq!(
            request_image_base64(Some(&messages)).unwrap().as_deref(),
            Some("YWJj")
        );
        let remote = serde_json::json!([{
            "role": "user",
            "content": [{ "type": "image_url", "image_url": { "url": "https://example/image.png" } }]
        }]);
        assert!(request_image_base64(Some(&remote))
            .unwrap_err()
            .to_string()
            .contains("remote"));
    }

    #[test]
    fn openai_messages_normalize_roles_content_and_tool_history() {
        let body = serde_json::json!({
            "messages": [
                { "role": "developer", "content": "system policy" },
                { "role": "user", "content": [
                    { "type": "text", "text": "first" },
                    { "type": "image_url", "image_url": { "url": "ignored" } },
                    { "type": "text", "text": " second" }
                ] },
                {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "tool reasoning",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"README.md\"}"
                        }
                    }]
                },
                { "role": "toolResult", "tool_call_id": "call_1", "content": "done" },
                { "role": "unsupported", "content": "drop me" }
            ]
        });
        // Flag OFF — today's exact shape unchanged, no reasoning_content key
        let normalized = normalize_openai_messages(body.get("messages"), false);
        assert_eq!(normalized.as_array().unwrap().len(), 4);
        assert_eq!(normalized[0]["role"], "system");
        assert_eq!(normalized[1]["content"], "first second");
        assert_eq!(normalized[2]["content"], "");
        assert_eq!(normalized[2]["tool_plan"], "tool reasoning");
        assert!(normalized[2].get("reasoning_content").is_none());
        assert_eq!(normalized[2]["tool_calls"][0]["name"], "read_file");
        assert_eq!(normalized[2]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            normalized[2]["tool_calls"][0]["arguments"],
            serde_json::json!({ "path": "README.md" })
        );
        assert_eq!(normalized[3]["role"], "tool");
        assert_eq!(normalized[3]["tool_call_id"], "call_1");
        // Flag ON — reasoning dual-written, content still visible-only
        let normalized_on = normalize_openai_messages(body.get("messages"), true);
        assert_eq!(normalized_on[2]["content"], "");
        assert_eq!(normalized_on[2]["tool_plan"], "tool reasoning");
        assert_eq!(normalized_on[2]["reasoning_content"], "tool reasoning");
        assert_eq!(
            normalized_on[2]["reasoning_content"],
            normalized_on[2]["tool_plan"]
        );
        assert_eq!(normalized_on[2]["tool_calls"][0]["id"], "call_1");
    }

    #[test]
    fn openai_assistant_history_strips_thinking_and_preserves_fallback_arguments() {
        let body = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "<think>private plan</think>\n\nvisible answer",
                "tool_calls": [{
                    "function": { "name": "broken", "arguments": "not-json" }
                }]
            }]
        });
        // Flag OFF — repair into _raw
        let normalized = normalize_openai_messages(body.get("messages"), false);
        assert_eq!(normalized[0]["content"], "visible answer");
        assert_eq!(normalized[0]["tool_plan"], "private plan");
        assert_eq!(
            normalized[0]["tool_calls"][0]["arguments"],
            serde_json::json!({ "_raw": "not-json" })
        );
        // Flag ON — string that does not parse to object is retained as string for daemon to reject
        let normalized_on = normalize_openai_messages(body.get("messages"), true);
        assert_eq!(normalized_on[0]["content"], "visible answer");
        assert_eq!(normalized_on[0]["tool_plan"], "private plan");
        assert_eq!(normalized_on[0]["reasoning_content"], "private plan");
        assert_eq!(
            normalized_on[0]["tool_calls"][0]["arguments"],
            serde_json::Value::String("not-json".into())
        );
    }

    #[test]
    fn jinja_started_think_routes_reasoning_then_visible_answer() {
        let mut router = ThinkChannelRouter::default();
        router.set_started_in_think(true);
        assert_eq!(
            router.push("reasoning body"),
            vec![ThinkFragment::Reasoning("reasoning body".into())]
        );
        assert!(router.push("</thi").is_empty());
        assert_eq!(
            router.push("nk>\n\nvisible answer"),
            vec![ThinkFragment::Content("visible answer".into())]
        );
        assert!(router.finish().is_empty());
    }

    #[test]
    fn plain_jinja_tail_keeps_output_in_content() {
        let mut router = ThinkChannelRouter::default();
        router.set_started_in_think(false);
        assert_eq!(
            router.push("direct answer"),
            vec![ThinkFragment::Content("direct answer".into())]
        );
    }

    #[test]
    fn model_literal_think_frames_route_consistently() {
        for family in ["qwen", "lfm", "minimax"] {
            let mut router = ThinkChannelRouter::default();
            router.set_started_in_think(true);
            let mut fragments = router.push(&format!("{family} reasoning</thi"));
            fragments.extend(router.push("nk>\n\nvisible<|im_"));
            fragments.extend(router.push("end|>"));
            fragments.extend(router.finish());
            assert_eq!(
                fragments,
                vec![
                    ThinkFragment::Reasoning(format!("{family} reasoning")),
                    ThinkFragment::Content("visible".into()),
                ],
                "{family}"
            );
        }
    }

    #[test]
    fn daemon_semantic_channels_override_literal_think_state() {
        for family in ["deepseek", "cohere"] {
            let mut router = ThinkChannelRouter::default();
            router.set_started_in_think(true);
            let mut fragments = router.push_semantic(&format!("{family} reason<|im_"), true);
            fragments.extend(router.push_semantic("end|>", true));
            fragments.extend(router.push("visible answer"));
            fragments.extend(router.finish());
            assert_eq!(
                fragments,
                vec![
                    ThinkFragment::Reasoning(format!("{family} reason")),
                    ThinkFragment::Content("visible answer".into()),
                ],
                "{family}"
            );
        }
    }

    #[test]
    fn output_router_removes_orphan_close_and_split_terminators() {
        let mut router = ThinkChannelRouter::default();
        let mut fragments = router.push("</thi");
        fragments.extend(router.push("nk>\n\nanswer<|endof"));
        fragments.extend(router.push("text|>tail"));
        fragments.extend(router.finish());
        assert_eq!(
            fragments,
            vec![
                ThinkFragment::Content("answer".into()),
                ThinkFragment::Content("tail".into()),
            ]
        );
    }

    #[test]
    fn request_sampling_omits_builtins_but_recovers_shadowed_registry_values() {
        let builtins = resolve(Vec::<NamedLayer>::new()).unwrap();
        assert_eq!(
            request_f64(&builtins, "generation.temperature", None).unwrap(),
            None
        );

        let mut registry = ConfigLayer::default();
        registry.set_cli("generation.temperature", "1.0").unwrap();
        registry.set_cli("generation.top_k", "40").unwrap();
        registry.set_cli("generation.min_p", "0.05").unwrap();
        registry
            .set_cli("generation.presence_penalty", "1.5")
            .unwrap();
        registry
            .set_cli("prompt.system", "registry identity")
            .unwrap();
        let mut global = ConfigLayer::default();
        global.set_cli("generation.temperature", "0.7").unwrap();
        global.set_cli("generation.top_k", "10").unwrap();
        global.set_cli("generation.min_p", "0.1").unwrap();
        global
            .set_cli("generation.presence_penalty", "0.5")
            .unwrap();
        global.set_cli("prompt.system", "global identity").unwrap();
        let resolved = resolve([
            NamedLayer {
                source: ConfigSource::RegistryModel {
                    tag: "qwen:test".into(),
                    revision: "v1".into(),
                },
                layer: registry,
            },
            NamedLayer {
                source: ConfigSource::GlobalUser {
                    path: PathBuf::from("config.toml"),
                },
                layer: global,
            },
        ])
        .unwrap();
        assert_eq!(
            request_f64(&resolved, "generation.temperature", None).unwrap(),
            Some(1.0)
        );
        assert_eq!(
            request_f64(&resolved, "generation.temperature", Some(0.25)).unwrap(),
            Some(0.25)
        );
        assert_eq!(
            request_u64(&resolved, "generation.top_k", None).unwrap(),
            Some(40)
        );
        assert_eq!(
            request_f64(&resolved, "generation.min_p", None).unwrap(),
            Some(0.05)
        );
        assert_eq!(
            request_f64(&resolved, "generation.presence_penalty", None).unwrap(),
            Some(1.5)
        );
        assert_eq!(
            request_string(&resolved, "prompt.system", None).unwrap(),
            Some("registry identity".into())
        );
        assert_eq!(
            request_string(&resolved, "prompt.system", Some("explicit".into())).unwrap(),
            Some("explicit".into())
        );
    }

    #[test]
    fn process_config_projects_only_explicit_arch_sensitive_config() {
        const NAME: &str = "HIPFIRE_FP16";
        let builtins = resolve(Vec::<NamedLayer>::new()).unwrap();
        let process = hipfire_config::ProcessConfig::from_resolved(&builtins).unwrap();
        assert_eq!(process.legacy_value(NAME), None);

        let mut global = ConfigLayer::default();
        global.set_cli("kernel.fp16", "false").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: PathBuf::from("config.toml"),
            },
            layer: global,
        }])
        .unwrap();
        let process = hipfire_config::ProcessConfig::from_resolved(&resolved).unwrap();
        assert_eq!(process.legacy_value(NAME).as_deref(), Some("0"));
    }

    #[test]
    fn process_config_projects_typed_scalar_and_variant_config() {
        const NAMES: &[&str] = &[
            "HIPFIRE_DEVICES",
            "HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB",
            "HIPFIRE_GEMV_ROWS",
            "HIPFIRE_LM_HEAD_F16",
        ];
        let mut global = ConfigLayer::default();
        global.set_cli("hardware.devices", "2,3").unwrap();
        global
            .set_cli("hardware.uniform_vram_tolerance_gb", "1.5")
            .unwrap();
        global.set_cli("diagnostic.kernel.gemv_rows", "4").unwrap();
        global.set_cli("kernel.lm_head_f16", "f32").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: PathBuf::from("config.toml"),
            },
            layer: global,
        }])
        .unwrap();
        let process = hipfire_config::ProcessConfig::from_resolved(&resolved).unwrap();
        assert_eq!(process.legacy_value(NAMES[0]).as_deref(), Some("2,3"));
        assert_eq!(process.legacy_value(NAMES[1]).as_deref(), Some("1.5"));
        assert_eq!(process.legacy_value(NAMES[2]).as_deref(), Some("4"));
        assert_eq!(process.legacy_value(NAMES[3]).as_deref(), Some("f32"));
    }

    #[test]
    fn http_reasoning_and_completion_metadata_match_native_contract() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let mut request = serde_json::json!({});
        apply_http_reasoning_request(
            &serde_json::json!({ "reasoning_effort": "high" }),
            &resolved,
            &mut request,
            false,
        )
        .unwrap();
        assert_eq!(request["reasoning_effort"], "high");
        assert_eq!(request["max_think_tokens"], 4096);

        let mut deepseek_uncapped = serde_json::json!({});
        apply_http_reasoning_request(
            &serde_json::json!({ "reasoning_effort": "max" }),
            &resolved,
            &mut deepseek_uncapped,
            true,
        )
        .unwrap();
        assert_eq!(deepseek_uncapped["reasoning_effort"], "max");
        assert_eq!(deepseek_uncapped["max_think_tokens"], 0);

        let mut deepseek_explicitly_capped = serde_json::json!({});
        apply_http_reasoning_request(
            &serde_json::json!({
                "reasoning_effort": "max",
                "max_think_tokens": 1234
            }),
            &resolved,
            &mut deepseek_explicitly_capped,
            true,
        )
        .unwrap();
        assert_eq!(deepseek_explicitly_capped["reasoning_effort"], "max");
        assert_eq!(deepseek_explicitly_capped["max_think_tokens"], 1234);

        let mut disabled = serde_json::json!({});
        apply_http_reasoning_request(
            &serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": false }
            }),
            &resolved,
            &mut disabled,
            false,
        )
        .unwrap();
        assert_eq!(disabled["reasoning_effort"], "none");

        let completion = Completion {
            id: "chatcmpl_test".into(),
            created: 7,
            model: "qwen:test".into(),
            content: "answer".into(),
            reasoning_content: "reason".into(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done: serde_json::json!({
                "prompt_tokens": 12,
                "tokens": 7,
                "cached_tokens": 4,
                "ttft_ms": 8.5,
                "decode_tok_s": 115.0,
                "finish_reason": "stop",
                "mtp_ngram": 3,
                "ngram_mod_windows": 5,
                "ngram_mod_drafts": 12,
                "ngram_mod_accepted": 9,
                "ngram_mod_accept_rate": 0.75,
                "mtp_windows": 4,
                "ar_windows": 2,
                "mtp_retired": true
            }),
        };
        let json = completion_json(&completion);
        assert_eq!(json["usage"]["total_tokens"], 19);
        assert_eq!(json["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
        assert_eq!(json["timings"]["decode_tok_s"], 115.0);
        assert_eq!(json["timings"]["mtp_ngram"], 3);
        assert_eq!(json["timings"]["ngram_mod_windows"], 5);
        assert_eq!(json["timings"]["ngram_mod_drafts"], 12);
        assert_eq!(json["timings"]["ngram_mod_accepted"], 9);
        assert_eq!(json["timings"]["ngram_mod_accept_rate"], 0.75);
        assert_eq!(json["timings"]["mtp_windows"], 4);
        assert_eq!(json["timings"]["ar_windows"], 2);
        assert_eq!(json["timings"]["mtp_retired"], true);
        assert_eq!(json["created"], 7);

        let qwen_cached = Completion {
            id: "chatcmpl_qwen_cached".into(),
            created: 8,
            model: "qwen:test".into(),
            content: "answer".into(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done: serde_json::json!({
                "prefill_tokens": 8,
                "cached_tokens": 12,
                "tokens": 7
            }),
        };
        let qwen_json = completion_json(&qwen_cached);
        assert_eq!(qwen_json["usage"]["prompt_tokens"], 20);
        assert_eq!(qwen_json["usage"]["total_tokens"], 27);
        assert_eq!(
            qwen_json["usage"]["prompt_tokens_details"]["cached_tokens"],
            12
        );

        let preserved = Completion {
            preserve_thinking: true,
            reasoning_content: "private chain".into(),
            ..qwen_cached
        };
        let preserved_json = completion_json(&preserved);
        assert_eq!(
            preserved_json["choices"][0]["message"]["content"],
            "<think>private chain</think>\nanswer"
        );
        assert!(preserved_json["choices"][0]["message"]
            .get("reasoning_content")
            .is_none());
    }

    #[test]
    fn apply_reasoning_request_accepts_medium_and_xhigh() {
        use hipfire_config::{resolve, ConfigLayer, ConfigSource, NamedLayer};
        use std::path::PathBuf;
        for effort in ["low", "medium", "xhigh", "high", "max"] {
            let mut layer = ConfigLayer::default();
            layer.set_cli("reasoning.effort", effort).unwrap();
            layer.set_cli("reasoning.max_tokens", "0").unwrap();
            let resolved = resolve([NamedLayer {
                source: ConfigSource::GlobalUser {
                    path: PathBuf::from("config.toml"),
                },
                layer,
            }])
            .unwrap();
            let mut req = serde_json::json!({});
            crate::apply_reasoning_request(&resolved, &mut req).unwrap();
            assert_eq!(
                req["reasoning_effort"], effort,
                "effort {effort} must pass through unchanged"
            );
        }
        // auto stays unset (no key), none disables
        let mut auto_layer = ConfigLayer::default();
        auto_layer.set_cli("reasoning.effort", "auto").unwrap();
        auto_layer.set_cli("reasoning.max_tokens", "0").unwrap();
        let auto_resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: PathBuf::from("config.toml"),
            },
            layer: auto_layer,
        }])
        .unwrap();
        let mut auto_req = serde_json::json!({});
        crate::apply_reasoning_request(&auto_resolved, &mut auto_req).unwrap();
        assert!(
            auto_req.get("reasoning_effort").is_none(),
            "auto must stay undefined"
        );

        let mut none_layer = ConfigLayer::default();
        none_layer.set_cli("reasoning.effort", "none").unwrap();
        none_layer.set_cli("reasoning.max_tokens", "0").unwrap();
        let none_resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: PathBuf::from("config.toml"),
            },
            layer: none_layer,
        }])
        .unwrap();
        let mut none_req = serde_json::json!({});
        crate::apply_reasoning_request(&none_resolved, &mut none_req).unwrap();
        assert_eq!(none_req["reasoning_effort"], "none");
        assert_eq!(none_req["max_think_tokens"], 1);
        assert_eq!(none_req["assistant_prefix"], "closed_think");
    }

    #[test]
    fn daemon_tool_calls_map_to_openai_shape() {
        let calls = vec![
            ToolCall {
                id: None,
                name: "read_file".into(),
                arguments: serde_json::json!({ "path": "README.md" }),
                rendered_body: None,
            },
            ToolCall {
                id: None,
                name: "write_file".into(),
                arguments: serde_json::json!({ "path": "out.txt", "text": "hi" }),
                rendered_body: None,
            },
        ];
        let mapped = openai_tool_calls(&calls);
        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0]["id"], "call_0");
        assert_eq!(mapped[1]["id"], "call_1");
        assert_eq!(mapped[0]["type"], "function");
        assert_eq!(mapped[0]["function"]["name"], "read_file");
        assert_eq!(
            mapped[0]["function"]["arguments"],
            serde_json::json!(r#"{"path":"README.md"}"#)
        );
        assert_eq!(mapped[1]["function"]["name"], "write_file");
    }

    #[test]
    fn completion_json_pure_tool_turn_uses_null_content() {
        let completion = sample_completion(
            "",
            vec![sample_tc(
                "read_file",
                serde_json::json!({ "path": "a.rs" }),
            )],
            "tool_calls",
        );
        let json = completion_json(&completion);
        assert!(json["choices"][0]["message"]["content"].is_null());
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_0"
        );
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
    }

    #[test]
    fn completion_json_preserves_daemon_length_without_calls() {
        // Fold withholds calls on length; serializer must not invent tool_calls.
        let completion = sample_completion("", Vec::new(), "length");
        let json = completion_json(&completion);
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        assert!(json["choices"][0]["message"].get("tool_calls").is_none());
        assert_eq!(json["choices"][0]["message"]["content"], "");
    }

    #[test]
    fn completion_json_never_overrides_length_error_cancel_when_calls_present() {
        // Defense in depth: even if tool_calls leaked onto Completion, daemon
        // finish_reason wins and must not be rewritten to tool_calls; calls stay off wire.
        for reason in ["length", "error", "cancelled", "aborted"] {
            let completion = sample_completion(
                "",
                vec![sample_tc("read_file", serde_json::json!({ "path": "x" }))],
                reason,
            );
            let json = completion_json(&completion);
            assert_eq!(
                json["choices"][0]["finish_reason"], reason,
                "must preserve daemon finish_reason={reason}"
            );
            assert!(
                json["choices"][0]["message"].get("tool_calls").is_none(),
                "{reason} must not expose message.tool_calls"
            );
            // empty content + withheld calls → empty string, not null pure-tool
            assert_eq!(json["choices"][0]["message"]["content"], "");
        }
    }

    #[test]
    fn completion_json_stop_text_has_string_content_no_tool_calls() {
        let completion = sample_completion("hello world", Vec::new(), "stop");
        let json = completion_json(&completion);
        assert_eq!(json["choices"][0]["message"]["content"], "hello world");
        assert!(json["choices"][0]["message"].get("tool_calls").is_none());
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn openai_stream_delta_forwards_only_clean_content_reasoning() {
        assert_eq!(
            openai_stream_delta_for_event(&serde_json::json!({
                "type": "token",
                "text": "hi"
            })),
            Some(serde_json::json!({ "content": "hi" }))
        );
        assert_eq!(
            openai_stream_delta_for_event(&serde_json::json!({
                "type": "reasoning",
                "text": "plan"
            })),
            Some(serde_json::json!({ "reasoning_content": "plan" }))
        );
        // Mid-stream tool_calls must never become an SSE delta.
        assert!(openai_stream_delta_for_event(&serde_json::json!({
            "type": "tool_calls",
            "calls": [{ "name": "read_file", "arguments": {} }]
        }))
        .is_none());
        assert!(openai_stream_delta_for_event(&serde_json::json!({
            "type": "done",
            "finish_reason": "stop"
        }))
        .is_none());
    }

    #[test]
    fn openai_stream_tool_safe_terminal_releases_calls_then_usage_then_done_shape() {
        let completion = sample_completion(
            "",
            vec![
                sample_tc("read_file", serde_json::json!({ "path": "a.rs" })),
                sample_tc("write_file", serde_json::json!({ "path": "b.rs" })),
            ],
            "tool_calls",
        );
        let chunks = openai_stream_terminal_chunks(&completion, true);
        assert_eq!(chunks.len(), 3, "tool delta + terminal + usage");

        // 1) tool_calls release with stable response-scoped ids/indices
        let tool_delta = &chunks[0]["choices"][0]["delta"]["tool_calls"];
        assert_eq!(tool_delta.as_array().map(|a| a.len()), Some(2));
        assert_eq!(tool_delta[0]["id"], "call_0");
        assert_eq!(tool_delta[0]["index"], 0);
        assert_eq!(tool_delta[1]["id"], "call_1");
        assert_eq!(tool_delta[1]["index"], 1);
        assert!(chunks[0]["choices"][0]["finish_reason"].is_null());
        assert!(chunks[0].get("usage").is_none());

        // 2) terminal choice with empty delta
        assert_eq!(chunks[1]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(chunks[1]["choices"][0]["delta"], serde_json::json!({}));
        assert!(
            chunks[1].get("usage").is_none(),
            "usage must not ride terminal"
        );

        // 3) separate choices:[] usage chunk
        assert_eq!(chunks[2]["choices"], serde_json::json!([]));
        assert_eq!(chunks[2]["usage"]["prompt_tokens"], 3);
        assert_eq!(chunks[2]["usage"]["completion_tokens"], 5);

        // Parity with non-stream ids/arguments
        let nonstream = completion_json(&completion);
        assert_eq!(
            nonstream["choices"][0]["message"]["tool_calls"][0]["id"],
            tool_delta[0]["id"]
        );
        assert_eq!(
            nonstream["choices"][0]["message"]["tool_calls"][0]["function"],
            tool_delta[0]["function"]
        );
        assert!(nonstream["choices"][0]["message"]["content"].is_null());
    }

    #[test]
    fn openai_stream_length_terminal_exposes_no_call_deltas() {
        let completion = sample_completion(
            "partial",
            // Even if present, non-tool-safe finish must not release.
            vec![sample_tc("read_file", serde_json::json!({ "path": "x" }))],
            "length",
        );
        let chunks = openai_stream_terminal_chunks(&completion, false);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0]["choices"][0]["finish_reason"], "length");
        assert!(chunks[0]["choices"][0]["delta"].get("tool_calls").is_none());
        assert!(chunks[0].get("usage").is_none());
    }

    #[test]
    fn openai_stream_include_usage_false_skips_usage_chunk() {
        let completion = sample_completion("ok", Vec::new(), "stop");
        let chunks = openai_stream_terminal_chunks(&completion, false);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0]["choices"][0]["finish_reason"], "stop");
        assert!(chunks[0].get("usage").is_none());
    }

    #[test]
    fn openai_stream_and_nonstream_paired_transcript_tool_safe() {
        // Paired transcript: fold-shaped Completion → both serializers agree.
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-7", 7);
        fold.push(&serde_json::json!({
            "type": "tool_calls",
            "calls": [
                { "name": "read_file", "arguments": { "path": "a" } },
                { "name": "write_file", "arguments": { "path": "b" } }
            ],
            "id": "req-7",
            "attempt_id": 7
        }))
        .unwrap();
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 4,
            "id": "req-7",
            "attempt_id": 7,
            "calls": [
                { "name": "read_file", "arguments": { "path": "a" } },
                { "name": "write_file", "arguments": { "path": "b" } }
            ]
        }))
        .unwrap();

        let completion = Completion {
            id: "chatcmpl_pair".into(),
            created: 99,
            model: "m".into(),
            content: fold.content().to_owned(),
            reasoning_content: fold.reasoning_content().to_owned(),
            preserve_thinking: false,
            tool_calls: fold.executable_tool_calls().to_vec(),
            done: fold.done().cloned().unwrap(),
        };

        let nonstream = completion_json(&completion);
        assert!(nonstream["choices"][0]["message"]["content"].is_null());
        assert_eq!(nonstream["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            nonstream["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_0"
        );
        assert_eq!(
            nonstream["choices"][0]["message"]["tool_calls"][1]["id"],
            "call_1"
        );

        // Mid-stream fold forward never includes tool_calls.
        assert!(openai_stream_delta_for_event(&serde_json::json!({
            "type": "tool_calls",
            "calls": fold.executable_tool_calls()
        }))
        .is_none());

        let stream_chunks = openai_stream_terminal_chunks(&completion, true);
        assert_eq!(
            stream_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
            nonstream["choices"][0]["message"]["tool_calls"][0]["id"]
        );
        assert_eq!(
            stream_chunks[0]["choices"][0]["delta"]["tool_calls"][1]["function"],
            nonstream["choices"][0]["message"]["tool_calls"][1]["function"]
        );
        assert_eq!(
            stream_chunks[1]["choices"][0]["finish_reason"],
            "tool_calls"
        );
        assert_eq!(stream_chunks[2]["choices"], serde_json::json!([]));
        assert!(stream_chunks[2].get("usage").is_some());
    }

    #[test]
    fn openai_stream_and_nonstream_paired_transcript_length_no_calls() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-8", 8);
        fold.push(&serde_json::json!({
            "type": "token",
            "text": "partial",
            "id": "req-8",
            "attempt_id": 8
        }))
        .unwrap();
        fold.push(&serde_json::json!({
            "type": "tool_calls",
            "calls": [{ "name": "read_file", "arguments": { "path": "x" } }],
            "id": "req-8",
            "attempt_id": 8
        }))
        .unwrap();
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "length",
            "id": "req-8",
            "attempt_id": 8
        }))
        .unwrap();

        let completion = Completion {
            id: "chatcmpl_len".into(),
            created: 1,
            model: "m".into(),
            content: fold.content().to_owned(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: fold.executable_tool_calls().to_vec(),
            done: fold.done().cloned().unwrap(),
        };
        assert!(completion.tool_calls.is_empty());

        let nonstream = completion_json(&completion);
        assert_eq!(nonstream["choices"][0]["finish_reason"], "length");
        assert!(nonstream["choices"][0]["message"]
            .get("tool_calls")
            .is_none());
        assert_eq!(nonstream["choices"][0]["message"]["content"], "partial");

        let stream_chunks = openai_stream_terminal_chunks(&completion, true);
        assert_eq!(stream_chunks.len(), 2); // terminal + usage, no tool release
        assert_eq!(stream_chunks[0]["choices"][0]["finish_reason"], "length");
        assert!(stream_chunks[0]["choices"][0]["delta"]
            .get("tool_calls")
            .is_none());
        assert_eq!(stream_chunks[1]["choices"], serde_json::json!([]));
    }

    #[test]
    fn openai_stream_and_nonstream_paired_missing_finish_suppresses_leaked_calls() {
        // Missing finish_reason must never synthesize tool_calls from buffered/leaked calls.
        let leaked = vec![sample_tc("read_file", serde_json::json!({ "path": "x" }))];
        let completion = sample_completion_with_done(
            "",
            leaked,
            serde_json::json!({
                "prompt_tokens": 3,
                "tokens": 5,
                "cached_tokens": 1,
                "tok_s": 10.0,
            }),
        );

        let nonstream = completion_json(&completion);
        assert_ne!(
            nonstream["choices"][0]["finish_reason"], "tool_calls",
            "missing finish_reason must not become tool_calls"
        );
        assert!(
            nonstream["choices"][0]["message"]
                .get("tool_calls")
                .is_none(),
            "missing finish_reason must suppress structured calls"
        );

        let stream_chunks = openai_stream_terminal_chunks(&completion, false);
        assert!(
            stream_chunks.iter().all(|c| c["choices"][0]
                .get("delta")
                .and_then(|d| d.get("tool_calls"))
                .is_none()),
            "stream must not release tool deltas without explicit tool_calls terminal"
        );
        assert_ne!(
            stream_chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    #[test]
    fn openai_stream_and_nonstream_paired_null_finish_suppresses_leaked_calls() {
        // Null finish_reason is not an explicit tool_calls terminal.
        let leaked = vec![sample_tc("read_file", serde_json::json!({ "path": "x" }))];
        let completion = sample_completion_with_done(
            "",
            leaked,
            serde_json::json!({
                "finish_reason": null,
                "prompt_tokens": 3,
                "tokens": 5,
                "cached_tokens": 1,
                "tok_s": 10.0,
            }),
        );

        let nonstream = completion_json(&completion);
        assert_ne!(
            nonstream["choices"][0]["finish_reason"], "tool_calls",
            "null finish_reason must not become tool_calls"
        );
        assert!(
            nonstream["choices"][0]["message"]
                .get("tool_calls")
                .is_none(),
            "null finish_reason must suppress structured calls"
        );

        let stream_chunks = openai_stream_terminal_chunks(&completion, false);
        assert!(
            stream_chunks.iter().all(|c| c["choices"][0]
                .get("delta")
                .and_then(|d| d.get("tool_calls"))
                .is_none()),
            "stream must not release tool deltas on null finish_reason"
        );
        assert_ne!(
            stream_chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    #[test]
    fn openai_stream_and_nonstream_paired_explicit_tool_calls_releases_calls() {
        // Only an explicit raw daemon finish_reason of tool_calls may expose calls.
        let calls = vec![sample_tc(
            "read_file",
            serde_json::json!({ "path": "a.rs" }),
        )];
        let completion = sample_completion("", calls, "tool_calls");

        let nonstream = completion_json(&completion);
        assert_eq!(nonstream["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            nonstream["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_0"
        );
        assert!(nonstream["choices"][0]["message"]["content"].is_null());

        let stream_chunks = openai_stream_terminal_chunks(&completion, false);
        assert_eq!(stream_chunks.len(), 2, "tool delta + terminal");
        assert_eq!(
            stream_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_0"
        );
        assert_eq!(
            stream_chunks[1]["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    #[test]
    fn semantic_fold_accumulates_content_and_reasoning_without_marker_parse() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-1", 1);
        // Classifier-authorized prose may quote protocol lexemes; fold must not
        // invent tool calls or strip them — only daemon tool_calls count.
        let forwarded = fold
            .push(&serde_json::json!({
                "type": "token",
                "text": "use <tool_call> as documentation",
                "id": "req-1",
                "attempt_id": 1
            }))
            .expect("token");
        assert_eq!(
            forwarded,
            vec![serde_json::json!({
                "type": "token",
                "text": "use <tool_call> as documentation"
            })]
        );
        let forwarded = fold
            .push(&serde_json::json!({
                "type": "reasoning",
                "text": "plan step",
                "id": "req-1",
                "attempt_id": 1
            }))
            .expect("reasoning");
        assert_eq!(
            forwarded,
            vec![serde_json::json!({ "type": "reasoning", "text": "plan step" })]
        );
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "stop",
            "id": "req-1",
            "attempt_id": 1
        }))
        .expect("done");
        assert_eq!(fold.content(), "use <tool_call> as documentation");
        assert_eq!(fold.reasoning_content(), "plan step");
        assert!(fold.executable_tool_calls().is_empty());
        assert_eq!(
            fold.done()
                .and_then(|d| d.get("finish_reason"))
                .and_then(|v| v.as_str()),
            Some("stop")
        );
    }

    #[test]
    fn semantic_fold_keeps_think_and_im_end_markers_verbatim_including_splits() {
        // Critical: v2 fold must never invoke ThinkChannelRouter / marker scan.
        // Recognized control literals must survive whole and across chunk boundaries.
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-42", 42);
        let pieces = ["<thi", "nk>plan</thi", "nk>\nanswer<|im_", "end|>tail"];
        let mut forwarded_text = String::new();
        for piece in pieces {
            let forwarded = fold
                .push(&serde_json::json!({
                    "type": "token",
                    "text": piece,
                    "id": "req-42",
                    "attempt_id": 42
                }))
                .expect("chunk");
            assert_eq!(forwarded.len(), 1);
            assert_eq!(forwarded[0]["type"], "token");
            assert_eq!(forwarded[0]["text"], piece);
            forwarded_text.push_str(piece);
        }
        let expected = "<think>plan</think>\nanswer<|im_end|>tail";
        assert_eq!(forwarded_text, expected);
        assert_eq!(fold.content(), expected);

        // Reasoning channel is also verbatim (no strip of </think> / <|im_end|>).
        fold.begin_attempt("req-43", 43);
        fold.push(&serde_json::json!({
            "type": "reasoning",
            "text": "r<think>x</think><|im_end|>",
            "id": "req-43",
            "attempt_id": 43
        }))
        .expect("reasoning markers");
        assert_eq!(fold.reasoning_content(), "r<think>x</think><|im_end|>");
        assert!(fold.content().is_empty());
    }

    #[test]
    fn semantic_fold_buffers_tool_calls_until_tool_safe_done() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-2", 2);
        let forwarded = fold
            .push(&serde_json::json!({
                "type": "token",
                "text": "calling",
                "id": "req-2",
                "attempt_id": 2
            }))
            .expect("token");
        assert_eq!(forwarded.len(), 1);
        let forwarded = fold
            .push(&serde_json::json!({
                "type": "tool_calls",
                "calls": [sample_tool_call("read_file")],
                "id": "req-2",
                "attempt_id": 2
            }))
            .expect("tool_calls");
        // Mid-stream: nothing forwarded; calls stay buffered and non-executable.
        assert!(forwarded.is_empty());
        assert_eq!(fold.buffered_tool_calls().len(), 1);
        assert!(fold.executable_tool_calls().is_empty());

        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "tool_calls",
            "tok_s": 12.5,
            "id": "req-2",
            "attempt_id": 2,
            "calls": [sample_tool_call("read_file")]
        }))
        .expect("done");
        assert_eq!(fold.executable_tool_calls().len(), 1);
        assert_eq!(fold.executable_tool_calls()[0].name, "read_file");
        assert_eq!(
            fold.done()
                .and_then(|d| d.get("finish_reason"))
                .and_then(|v| v.as_str()),
            Some("tool_calls")
        );
        // Daemon finish_reason preserved verbatim (no fold-side rewrite).
        assert_eq!(
            fold.done()
                .and_then(|d| d.get("tok_s"))
                .and_then(|v| v.as_f64()),
            Some(12.5)
        );
    }

    #[test]
    fn semantic_fold_length_terminal_exposes_no_executable_calls() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-3", 3);
        fold.push(&serde_json::json!({
            "type": "tool_calls",
            "calls": [sample_tool_call("write_file"), sample_tool_call("read_file")],
            "id": "req-3",
            "attempt_id": 3
        }))
        .expect("buffered");
        assert_eq!(fold.buffered_tool_calls().len(), 2);
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "length",
            "id": "req-3",
            "attempt_id": 3
        }))
        .expect("done");
        assert!(
            fold.executable_tool_calls().is_empty(),
            "length must never release buffered calls"
        );
        assert_eq!(
            fold.done()
                .and_then(|d| d.get("finish_reason"))
                .and_then(|v| v.as_str()),
            Some("length"),
            "daemon finish_reason must not be rewritten to tool_calls"
        );
    }

    #[test]
    fn semantic_fold_error_and_abort_terminals_expose_no_calls() {
        for reason in ["error", "aborted", "cancelled"] {
            let mut fold = SemanticEventFold::new();
            fold.begin_attempt("req-4", 4);
            fold.push(&serde_json::json!({
                "type": "tool_calls",
                "calls": [sample_tool_call("read_file")],
                "id": "req-4",
                "attempt_id": 4
            }))
            .expect("buffered");
            fold.push(&serde_json::json!({
                "type": "done",
                "finish_reason": reason,
                "id": "req-4",
                "attempt_id": 4
            }))
            .expect("done");
            assert!(
                fold.executable_tool_calls().is_empty(),
                "{reason} must not expose executable calls"
            );
            assert_eq!(
                fold.done()
                    .and_then(|d| d.get("finish_reason"))
                    .and_then(|v| v.as_str()),
                Some(reason)
            );
        }
    }

    #[test]
    fn semantic_fold_stop_with_empty_buffer_stays_empty() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-5", 5);
        fold.push(&serde_json::json!({
            "type": "token",
            "text": "hello",
            "id": "req-5",
            "attempt_id": 5
        }))
        .expect("token");
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "stop",
            "id": "req-5",
            "attempt_id": 5
        }))
        .expect("done");
        assert!(fold.executable_tool_calls().is_empty());
        assert_eq!(fold.content(), "hello");
    }

    #[test]
    fn semantic_fold_rejects_stale_attempt_events() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-10", 10);
        fold.push(&serde_json::json!({
            "type": "token",
            "text": "a1",
            "id": "req-10",
            "attempt_id": 10
        }))
        .expect("current");
        let err = fold
            .push(&serde_json::json!({
                "type": "token",
                "text": "stale",
                "id": "req-10",
                "attempt_id": 9
            }))
            .expect_err("stale attempt");
        assert_eq!(
            err,
            SemanticFoldError::StaleAttempt {
                current: 10,
                got: 9
            }
        );
        // Current attempt state must remain intact after rejection.
        assert_eq!(fold.content(), "a1");
        assert_eq!(fold.current_attempt_id(), Some(10));
    }

    #[test]
    fn semantic_fold_rejects_missing_malformed_from_first_event() {
        // Critical: after begin_attempt, every event including the first must
        // carry a matching numeric attempt_id. No lazy correlation / uncorrelated
        // stream acceptance.
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-11", 11);

        let missing = fold
            .push(&serde_json::json!({
                "type": "token",
                "text": "no id",
                "id": "req-11"
            }))
            .expect_err("missing attempt_id");
        assert_eq!(missing, SemanticFoldError::MissingAttemptId { current: 11 });
        assert!(fold.content().is_empty());

        let malformed_string = fold
            .push(&serde_json::json!({
                "type": "token",
                "text": "bad",
                "id": "req-11",
                "attempt_id": "11"
            }))
            .expect_err("string attempt_id");
        assert_eq!(
            malformed_string,
            SemanticFoldError::MalformedAttemptId { current: 11 }
        );

        let malformed_null = fold
            .push(&serde_json::json!({
                "type": "gen_start",
                "id": "req-11",
                "attempt_id": null
            }))
            .expect_err("null attempt_id");
        assert_eq!(
            malformed_null,
            SemanticFoldError::MalformedAttemptId { current: 11 }
        );

        // Without begin_attempt, push fails closed (no uncorrelated stream).
        let mut cold = SemanticEventFold::new();
        let err = cold
            .push(&serde_json::json!({
                "type": "token",
                "text": "uncorrelated",
                "id": "req-1",
                "attempt_id": 1
            }))
            .expect_err("no active attempt");
        assert_eq!(err, SemanticFoldError::NoActiveAttempt);
    }

    #[test]
    fn semantic_fold_begin_attempt_clears_attempt_local_state() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-1", 1);
        fold.push(&serde_json::json!({
            "type": "token",
            "text": "first",
            "id": "req-1",
            "attempt_id": 1
        }))
        .expect("token");
        fold.push(&serde_json::json!({
            "type": "reasoning",
            "text": "think1",
            "id": "req-1",
            "attempt_id": 1
        }))
        .expect("reasoning");
        fold.push(&serde_json::json!({
            "type": "tool_calls",
            "calls": [sample_tool_call("read_file")],
            "id": "req-1",
            "attempt_id": 1
        }))
        .expect("calls");
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "tool_calls",
            "id": "req-1",
            "attempt_id": 1,
            "calls": [sample_tool_call("read_file")]
        }))
        .expect("done");
        assert!(!fold.content().is_empty());
        assert!(!fold.executable_tool_calls().is_empty());

        fold.begin_attempt("req-2", 2);
        assert_eq!(fold.current_attempt_id(), Some(2));
        assert!(fold.content().is_empty());
        assert!(fold.reasoning_content().is_empty());
        assert!(fold.buffered_tool_calls().is_empty());
        assert!(fold.executable_tool_calls().is_empty());
        assert!(fold.done().is_none());

        fold.push(&serde_json::json!({
            "type": "token",
            "text": "second",
            "id": "req-2",
            "attempt_id": 2
        }))
        .expect("retry token");
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "stop",
            "id": "req-2",
            "attempt_id": 2
        }))
        .expect("retry done");
        assert_eq!(fold.content(), "second");
        assert!(fold.executable_tool_calls().is_empty());
        assert_eq!(
            fold.done()
                .and_then(|d| d.get("finish_reason"))
                .and_then(|v| v.as_str()),
            Some("stop")
        );
    }

    #[test]
    fn producer_to_fold_contract_v2_verbatim_legacy_outside() {
        // Important: gen_start.contract_version == 2 selects SemanticEventFold
        // (verbatim text). Legacy non-tool raw-think stays on ThinkChannelRouter
        // outside the fold — proved here without inventing a second fold path.

        // --- v2 producer path (fold) ---
        let mut v2 = SemanticEventFold::new();
        v2.begin_attempt("req-100", 100);
        v2.push(&serde_json::json!({
            "type": "gen_start",
            "contract_version": 2,
            "started_in_think": true,
            "id": "req-100",
            "attempt_id": 100
        }))
        .expect("v2 gen_start");
        // started_in_think must not open a think channel inside the fold.
        let text = "visible <think>not-routed</think> <|im_end|>";
        let fwd = v2
            .push(&serde_json::json!({
                "type": "token",
                "text": text,
                "id": "req-100",
                "attempt_id": 100
            }))
            .expect("v2 token");
        assert_eq!(
            fwd,
            vec![serde_json::json!({ "type": "token", "text": text })]
        );
        assert_eq!(v2.content(), text);
        assert!(v2.reasoning_content().is_empty());

        // --- legacy producer path (ThinkChannelRouter only; outside fold) ---
        let mut router = ThinkChannelRouter::default();
        router.set_started_in_think(true);
        let mut fragments = router.push("legacy reason</thi");
        fragments.extend(router.push("nk>\n\nlegacy answer"));
        fragments.extend(router.finish());
        assert_eq!(
            fragments,
            vec![
                ThinkFragment::Reasoning("legacy reason".into()),
                ThinkFragment::Content("legacy answer".into()),
            ],
            "legacy MiniMax/Cohere-style markers still route outside SemanticEventFold"
        );

        // Fold must not be the home of that routing: pushing the same bytes
        // through a correlated fold keeps them as content, not reasoning split.
        let mut not_legacy = SemanticEventFold::new();
        not_legacy.begin_attempt("req-101", 101);
        not_legacy
            .push(&serde_json::json!({
                "type": "token",
                "text": "legacy reason</think>\n\nlegacy answer",
                "id": "req-101",
                "attempt_id": 101
            }))
            .expect("fold token");
        assert_eq!(
            not_legacy.content(),
            "legacy reason</think>\n\nlegacy answer"
        );
        assert!(not_legacy.reasoning_content().is_empty());
    }

    #[test]
    fn next_attempt_id_is_nonzero_and_monotonic() {
        let a = next_attempt_id();
        let b = next_attempt_id();
        assert_ne!(a, 0);
        assert_ne!(b, 0);
        assert!(b > a);
    }

    #[test]
    fn task15_attempt_latches_truth_table() {
        let cases: &[(&str, bool, bool)] = &[
            ("token", true, false),
            ("reasoning", true, false),
            ("commit_ready", false, true),
            ("tool_calls", false, false),
            ("gen_start", false, false),
            ("done", false, false),
            ("error", false, false),
        ];
        for (ty, want_visible, want_commit) in cases {
            let mut latches = AttemptLatches::default();
            latches.observe(&serde_json::json!({ "type": ty }));
            assert_eq!(latches.visible, *want_visible, "visible for {ty}");
            assert_eq!(
                latches.commit_ready_seen, *want_commit,
                "commit_ready_seen for {ty}"
            );
        }
    }

    #[test]
    fn task15_decide_retry_classifier_truth_table() {
        let aid = 42u64;
        let clean = AttemptLatches::default();
        let mut visible = AttemptLatches::default();
        visible.visible = true;
        let mut committed = AttemptLatches::default();
        committed.commit_ready_seen = true;

        let ok_err = task15_daemon_err(hipfire_client::error_class::TRANSIENT, true, aid);
        assert_eq!(
            decide_retry(&ok_err, aid, &clean, true, true, 1),
            RetryDecision::Retry
        );

        let denials: &[(&str, RetryDecision)] = &[
            (
                "gate_off",
                decide_retry(&ok_err, aid, &clean, true, false, 1),
            ),
            (
                "attempt_2",
                decide_retry(&ok_err, aid, &clean, true, true, 2),
            ),
            (
                "visible",
                decide_retry(&ok_err, aid, &visible, true, true, 1),
            ),
            (
                "commit_ready",
                decide_retry(&ok_err, aid, &committed, true, true, 1),
            ),
            (
                "ineligible",
                decide_retry(&ok_err, aid, &clean, false, true, 1),
            ),
            (
                "attempt_mismatch",
                decide_retry(&ok_err, aid + 1, &clean, true, true, 1),
            ),
            (
                "not_retryable",
                decide_retry(
                    &task15_daemon_err(hipfire_client::error_class::TRANSIENT, false, aid),
                    aid,
                    &clean,
                    true,
                    true,
                    1,
                ),
            ),
            (
                "class_validation",
                decide_retry(
                    &task15_daemon_err(hipfire_client::error_class::VALIDATION, true, aid),
                    aid,
                    &clean,
                    true,
                    true,
                    1,
                ),
            ),
            (
                "class_malformed",
                decide_retry(
                    &task15_daemon_err(hipfire_client::error_class::MALFORMED, true, aid),
                    aid,
                    &clean,
                    true,
                    true,
                    1,
                ),
            ),
            (
                "class_internal",
                decide_retry(
                    &task15_daemon_err(hipfire_client::error_class::INTERNAL, true, aid),
                    aid,
                    &clean,
                    true,
                    true,
                    1,
                ),
            ),
            (
                "class_cancel",
                decide_retry(
                    &task15_daemon_err(hipfire_client::error_class::CANCEL, true, aid),
                    aid,
                    &clean,
                    true,
                    true,
                    1,
                ),
            ),
            (
                "non_daemon",
                decide_retry(&anyhow::anyhow!("plain error"), aid, &clean, true, true, 1),
            ),
            (
                "protocol",
                decide_retry(
                    &anyhow::Error::new(hipfire_client::ClientError::Protocol("x".into())),
                    aid,
                    &clean,
                    true,
                    true,
                    1,
                ),
            ),
        ];
        for (name, decision) in denials {
            assert_eq!(*decision, RetryDecision::Fail, "expected Fail for {name}");
        }
    }

    #[test]
    fn complete_request_fold_rejects_pre_start_token() {
        // Critical: no unchecked legacy default before gen_start.
        let err = fold_complete_request_stream(
            "req-7",
            7,
            &[serde_json::json!({
                "type": "token",
                "text": "too early",
                "id": "req-7",
                "attempt_id": 7
            })],
        )
        .expect_err("pre-start token must fail closed");
        assert_eq!(
            err,
            StreamContractError::PreStartEvent {
                event_type: "token".into()
            }
        );
    }

    #[test]
    fn complete_request_fold_rejects_missing_id_gen_start() {
        // Correlation before contract_version latch — missing id never selects legacy/v2.
        let err = fold_complete_request_stream(
            "req-7",
            7,
            &[serde_json::json!({
                "type": "gen_start",
                "contract_version": 2
            })],
        )
        .expect_err("missing request id on gen_start");
        assert_eq!(
            err,
            StreamContractError::MissingRequestId {
                expected: "req-7".into()
            }
        );

        let err_missing_attempt = fold_complete_request_stream(
            "req-7",
            7,
            &[serde_json::json!({
                "type": "gen_start",
                "contract_version": 2,
                "id": "req-7"
            })],
        )
        .expect_err("missing attempt_id on gen_start");
        assert_eq!(
            err_missing_attempt,
            StreamContractError::MissingAttemptId { expected: 7 }
        );

        let err_malformed = fold_complete_request_stream(
            "req-7",
            7,
            &[serde_json::json!({
                "type": "gen_start",
                "contract_version": 2,
                "id": "req-7",
                "attempt_id": "7"
            })],
        )
        .expect_err("string attempt_id on gen_start");
        assert_eq!(
            err_malformed,
            StreamContractError::MalformedAttemptId { expected: 7 }
        );

        let err_stale_first = fold_complete_request_stream(
            "req-7",
            7,
            &[serde_json::json!({
                "type": "gen_start",
                "contract_version": 2,
                "id": "req-7",
                "attempt_id": 99
            })],
        )
        .expect_err("stale attempt_id on first gen_start");
        assert_eq!(
            err_stale_first,
            StreamContractError::StaleAttempt {
                expected: 7,
                got: 99
            }
        );
    }

    #[test]
    fn complete_request_fold_rejects_stale_second_start_downgrade() {
        // Critical: after v2 is latched, a later stale/missing-id gen_start must
        // NOT downgrade the stream to legacy or re-latch contract_version.
        let events = [
            serde_json::json!({
                "type": "gen_start",
                "contract_version": 2,
                "id": "req-7",
                "attempt_id": 7
            }),
            serde_json::json!({
                "type": "token",
                "text": "kept",
                "id": "req-7",
                "attempt_id": 7
            }),
            // Stale second start — previously could flip contract_v2 = Some(false).
            serde_json::json!({
                "type": "gen_start",
                "id": "req-1",
                "attempt_id": 1
            }),
            serde_json::json!({
                "type": "token",
                "text": "would-be-legacy",
                "id": "req-7",
                "attempt_id": 7
            }),
        ];
        let err = fold_complete_request_stream("req-7", 7, &events)
            .expect_err("second gen_start must reject without downgrade");
        assert_eq!(err, StreamContractError::SecondGenStart);

        // Gate unit: v2 stays latched even if observe is retried after error.
        let mut gate = StreamContractGate::new("req-7", 7);
        assert_eq!(gate.observe(&events[0]).expect("first"), StreamContract::V2);
        assert_eq!(gate.observe(&events[1]).expect("token"), StreamContract::V2);
        assert_eq!(
            gate.observe(&events[2]).expect_err("second start"),
            StreamContractError::SecondGenStart
        );
        assert_eq!(gate.contract(), Some(StreamContract::V2));
        assert!(gate.is_v2());
        // Subsequent non-start events would still be v2 if caller continued —
        // production aborts the generate callback on SecondGenStart instead.
        assert_eq!(
            gate.observe(&events[3]).expect("still v2"),
            StreamContract::V2
        );

        // Missing-id second start is also SecondGenStart (never re-reads version).
        let mut gate2 = StreamContractGate::new("req-7", 7);
        gate2
            .observe(&serde_json::json!({
                "type": "gen_start",
                "contract_version": 2,
                "id": "req-7",
                "attempt_id": 7
            }))
            .unwrap();
        assert_eq!(
            gate2
                .observe(&serde_json::json!({
                    "type": "gen_start",
                    "contract_version": 1
                }))
                .expect_err("missing-id second start"),
            StreamContractError::SecondGenStart
        );
        assert_eq!(gate2.contract(), Some(StreamContract::V2));
    }

    #[test]
    fn complete_request_fold_valid_legacy_and_v2_starts() {
        // Valid v2 start → SemanticEventFold verbatim path.
        let v2 = fold_complete_request_stream(
            "req-42",
            42,
            &[
                serde_json::json!({
                    "type": "gen_start",
                    "contract_version": 2,
                    "started_in_think": true,
                    "id": "req-42",
                    "attempt_id": 42
                }),
                serde_json::json!({
                    "type": "token",
                    "text": "hi <think>raw</think>",
                    "id": "req-42",
                    "attempt_id": 42
                }),
                serde_json::json!({
                    "type": "done",
                    "finish_reason": "stop",
                    "id": "req-42",
                    "attempt_id": 42
                }),
            ],
        )
        .expect("valid v2 stream");
        assert_eq!(v2.contract, StreamContract::V2);
        assert_eq!(v2.content, "hi <think>raw</think>");
        assert!(v2.reasoning_content.is_empty());
        assert!(v2.tool_calls.is_empty());
        assert_eq!(
            v2.done
                .as_ref()
                .and_then(|d| d.get("finish_reason"))
                .and_then(|v| v.as_str()),
            Some("stop")
        );

        // Valid legacy start (missing/other contract_version) → ThinkChannelRouter.
        let legacy = fold_complete_request_stream(
            "req-42",
            42,
            &[
                serde_json::json!({
                    "type": "gen_start",
                    "started_in_think": true,
                    "id": "req-42",
                    "attempt_id": 42
                }),
                serde_json::json!({
                    "type": "token",
                    "text": "plan</think>\n\nanswer",
                    "id": "req-42",
                    "attempt_id": 42
                }),
                serde_json::json!({
                    "type": "done",
                    "finish_reason": "stop",
                    "id": "req-42",
                    "attempt_id": 42
                }),
            ],
        )
        .expect("valid legacy stream");
        assert_eq!(legacy.contract, StreamContract::Legacy);
        assert_eq!(legacy.reasoning_content, "plan");
        assert_eq!(legacy.content, "answer");

        // Explicit non-2 contract_version is also legacy.
        let legacy_v1 = fold_complete_request_stream(
            "req-3",
            3,
            &[
                serde_json::json!({
                    "type": "gen_start",
                    "contract_version": 1,
                    "id": "req-3",
                    "attempt_id": 3
                }),
                serde_json::json!({
                    "type": "token",
                    "text": "plain",
                    "id": "req-3",
                    "attempt_id": 3
                }),
                serde_json::json!({
                    "type": "done",
                    "finish_reason": "stop",
                    "id": "req-3",
                    "attempt_id": 3
                }),
            ],
        )
        .expect("contract_version 1 is legacy");
        assert_eq!(legacy_v1.contract, StreamContract::Legacy);
        assert_eq!(legacy_v1.content, "plain");
    }

    #[test]
    fn openai_adapter_preserves_names_and_nested_arguments() {
        let calls = vec![
            sample_tc(
                "search",
                serde_json::json!({
                    "query": "hipfire",
                    "filters": { "lang": ["rust", "c"], "limit": 3 },
                    "opts": { "nested": { "deep": true } }
                }),
            ),
            sample_tc("ping", serde_json::json!({})),
        ];
        let adapted = openai_tool_call_adapter_results(&calls);
        assert_eq!(adapted.len(), 2);
        assert_eq!(adapted[0].name, "search");
        assert_eq!(adapted[1].name, "ping");

        let args0: serde_json::Value =
            serde_json::from_str(&adapted[0].arguments).expect("args json");
        assert_eq!(args0["filters"]["lang"][1], "c");
        assert_eq!(args0["opts"]["nested"]["deep"], true);
        assert_eq!(adapted[1].arguments, "{}");

        let lowered = openai_tool_calls(&calls);
        assert_eq!(lowered[0]["function"]["name"], "search");
        assert_eq!(
            lowered[0]["function"]["arguments"].as_str().unwrap(),
            adapted[0].arguments
        );
    }

    #[test]
    fn openai_adapter_deterministic_stable_ids_and_indices() {
        let calls = vec![
            sample_tc("a", serde_json::json!({"n": 1})),
            sample_tc("b", serde_json::json!({"n": 2})),
            sample_tc("c", serde_json::json!({"n": 3})),
        ];
        let first = openai_tool_call_adapter_results(&calls);
        let second = openai_tool_call_adapter_results(&calls);
        assert_eq!(first, second, "adapter result must be deterministic");
        for (i, row) in first.iter().enumerate() {
            assert_eq!(row.index, i);
            assert_eq!(row.id, format!("call_{i}"));
            assert_eq!(row.name, calls[i].name);
        }

        let stream_delta = openai_tool_call_delta_from_adapter(&first);
        let nonstream = openai_tool_calls_from_adapter(&first);
        for i in 0..3 {
            assert_eq!(stream_delta["tool_calls"][i]["index"], i);
            assert_eq!(stream_delta["tool_calls"][i]["id"], format!("call_{i}"));
            assert_eq!(nonstream[i]["id"], format!("call_{i}"));
            assert!(
                nonstream[i].get("index").is_none(),
                "non-stream message.tool_calls must not carry stream index"
            );
        }
    }

    #[test]
    fn openai_stream_and_nonstream_share_one_adapter_result() {
        let calls = vec![
            sample_tc(
                "read_file",
                serde_json::json!({ "path": "a.rs", "meta": { "k": "v" } }),
            ),
            sample_tc("write_file", serde_json::json!({ "path": "b.rs" })),
        ];
        let adapted = openai_tool_call_adapter_results(&calls);
        let completion = sample_completion("", calls.clone(), "tool_calls");

        let nonstream = completion_json(&completion);
        let stream = openai_stream_terminal_chunks(&completion, false);
        let ns_calls = nonstream["choices"][0]["message"]["tool_calls"]
            .as_array()
            .expect("nonstream tool_calls");
        let st_calls = stream[0]["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .expect("stream tool_calls");

        assert_eq!(ns_calls.len(), adapted.len());
        assert_eq!(st_calls.len(), adapted.len());
        for (i, row) in adapted.iter().enumerate() {
            assert_eq!(ns_calls[i]["id"], row.id);
            assert_eq!(ns_calls[i]["function"]["name"], row.name);
            assert_eq!(ns_calls[i]["function"]["arguments"], row.arguments);
            assert_eq!(st_calls[i]["id"], row.id);
            assert_eq!(st_calls[i]["index"], row.index);
            assert_eq!(st_calls[i]["function"]["name"], row.name);
            assert_eq!(st_calls[i]["function"]["arguments"], row.arguments);
            assert_eq!(ns_calls[i]["function"], st_calls[i]["function"]);
            assert_eq!(ns_calls[i]["id"], st_calls[i]["id"]);
        }
    }

    #[test]
    fn openai_pure_tool_turn_content_is_null_not_empty_string() {
        let completion = sample_completion(
            "",
            vec![sample_tc("read_file", serde_json::json!({ "path": "x" }))],
            "tool_calls",
        );
        let json = completion_json(&completion);
        assert!(json["choices"][0]["message"]["content"].is_null());
        assert!(json["choices"][0]["message"]["tool_calls"].is_array());
    }

    #[test]
    fn openai_mixed_prose_and_calls_retains_prose_content() {
        let completion = sample_completion(
            "I'll look that up.",
            vec![sample_tc("search", serde_json::json!({ "q": "docs" }))],
            "tool_calls",
        );
        let json = completion_json(&completion);
        assert_eq!(
            json["choices"][0]["message"]["content"],
            "I'll look that up."
        );
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "search"
        );
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");

        let chunks = openai_stream_terminal_chunks(&completion, false);
        assert_eq!(
            chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "search"
        );
        assert_eq!(chunks[1]["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn openai_length_error_cancel_malformed_never_release_calls() {
        let leaked = vec![sample_tc("read_file", serde_json::json!({ "path": "x" }))];
        for reason in [
            "length",
            "error",
            "cancelled",
            "aborted",
            "malformed_protocol",
        ] {
            let completion = sample_completion("partial", leaked.clone(), reason);
            let nonstream = completion_json(&completion);
            assert_eq!(
                nonstream["choices"][0]["finish_reason"], reason,
                "{reason}: finish_reason must stay authoritative"
            );
            assert!(
                nonstream["choices"][0]["message"]
                    .get("tool_calls")
                    .is_none(),
                "{reason}: must not release message.tool_calls"
            );
            assert_eq!(nonstream["choices"][0]["message"]["content"], "partial");

            let stream = openai_stream_terminal_chunks(&completion, false);
            assert!(
                stream.iter().all(|c| {
                    c["choices"]
                        .as_array()
                        .and_then(|choices| choices.first())
                        .and_then(|ch| ch.get("delta"))
                        .and_then(|d| d.get("tool_calls"))
                        .is_none()
                }),
                "{reason}: stream must not emit tool_call deltas"
            );
            assert_eq!(stream[0]["choices"][0]["finish_reason"], reason);
        }
    }

    #[test]
    fn malformed_daemon_call_fails_at_canonical_boundary() {
        let err = tool_call_from_canonical_value(&serde_json::json!("not-an-object"))
            .expect_err("non-object");
        assert!(
            err.contains("JSON object") || err.contains("object"),
            "detail={err}"
        );

        let err = tool_call_from_canonical_value(&serde_json::json!({
            "arguments": { "path": "x" }
        }))
        .expect_err("missing name");
        assert!(err.contains("name"), "detail={err}");

        let err = tool_call_from_canonical_value(&serde_json::json!({
            "name": "   ",
            "arguments": {}
        }))
        .expect_err("empty name");
        assert!(err.contains("name"), "detail={err}");

        let legacy_err = tool_call_from_legacy_value(&serde_json::json!(null)).expect_err("null");
        assert!(!legacy_err.is_empty());

        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-11", 11);
        let fold_err = fold
            .push(&serde_json::json!({
                "type": "tool_calls",
                "calls": [{ "arguments": { "path": "x" } }],
                "id": "req-11",
                "attempt_id": 11
            }))
            .expect_err("malformed call must fail fold push");
        match fold_err {
            SemanticFoldError::MalformedToolCall { detail } => {
                assert!(detail.contains("name"), "detail={detail}");
            }
            other => panic!("expected MalformedToolCall, got {other:?}"),
        }
        assert!(
            fold.buffered_tool_calls().is_empty(),
            "malformed must not buffer a call"
        );
        assert!(fold.executable_tool_calls().is_empty());
    }

    #[test]
    fn semantic_fold_missing_calls_fails_closed_before_tool_terminal() {
        // Canonical v2 tool_calls without `calls` must not succeed the fold or
        // lower to finish_reason=tool_calls via the production stream boundary.
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-21", 21);
        let err = fold
            .push(&serde_json::json!({
                "type": "tool_calls",
                "id": "req-21",
                "attempt_id": 21
            }))
            .expect_err("missing calls must fail fold push");
        match err {
            SemanticFoldError::MalformedToolCall { detail } => {
                assert!(
                    detail.contains("calls") && detail.contains("array"),
                    "detail={detail}"
                );
            }
            other => panic!("expected MalformedToolCall, got {other:?}"),
        }
        assert!(fold.buffered_tool_calls().is_empty());
        assert!(fold.executable_tool_calls().is_empty());
        assert!(fold.done().is_none());

        let stream_err = fold_complete_request_stream(
            "req-21",
            21,
            &[
                serde_json::json!({
                    "type": "gen_start",
                    "id": "req-21",
                    "attempt_id": 21,
                    "contract_version": 2
                }),
                serde_json::json!({
                    "type": "tool_calls",
                    "id": "req-21",
                    "attempt_id": 21
                }),
                serde_json::json!({
                    "type": "done",
                    "finish_reason": "tool_calls",
                    "id": "req-21",
                    "attempt_id": 21
                }),
            ],
        )
        .expect_err("missing calls must fail complete_request fold boundary");
        match stream_err {
            StreamContractError::MalformedToolCall { detail } => {
                assert!(
                    detail.contains("calls") && detail.contains("array"),
                    "detail={detail}"
                );
            }
            other => panic!("expected StreamContractError::MalformedToolCall, got {other:?}"),
        }
    }

    #[test]
    fn semantic_fold_non_array_calls_fails_closed_before_tool_terminal() {
        // null / object / string `calls` are not arrays — fail closed before any
        // successful tool_calls terminal lowering can be produced.
        for calls in [
            serde_json::Value::Null,
            serde_json::json!({ "name": "read_file" }),
            serde_json::json!("read_file"),
        ] {
            let mut fold = SemanticEventFold::new();
            fold.begin_attempt("req-22", 22);
            let err = fold
                .push(&serde_json::json!({
                    "type": "tool_calls",
                    "calls": calls,
                    "id": "req-22",
                    "attempt_id": 22
                }))
                .expect_err("non-array calls must fail fold push");
            match err {
                SemanticFoldError::MalformedToolCall { detail } => {
                    assert!(
                        detail.contains("calls") && detail.contains("array"),
                        "detail={detail}"
                    );
                }
                other => panic!("expected MalformedToolCall, got {other:?}"),
            }
            assert!(fold.buffered_tool_calls().is_empty());
            assert!(fold.executable_tool_calls().is_empty());
            assert!(fold.done().is_none());
        }

        let stream_err = fold_complete_request_stream(
            "req-22",
            22,
            &[
                serde_json::json!({
                    "type": "gen_start",
                    "id": "req-22",
                    "attempt_id": 22,
                    "contract_version": 2
                }),
                serde_json::json!({
                    "type": "tool_calls",
                    "calls": null,
                    "id": "req-22",
                    "attempt_id": 22
                }),
                serde_json::json!({
                    "type": "done",
                    "finish_reason": "tool_calls",
                    "id": "req-22",
                    "attempt_id": 22
                }),
            ],
        )
        .expect_err("null calls must fail complete_request fold boundary");
        match stream_err {
            StreamContractError::MalformedToolCall { detail } => {
                assert!(
                    detail.contains("calls") && detail.contains("array"),
                    "detail={detail}"
                );
            }
            other => panic!("expected StreamContractError::MalformedToolCall, got {other:?}"),
        }
    }

    #[test]
    fn missing_or_lossy_endpoint_adapter_rejects_before_mutation() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mutations = AtomicUsize::new(0);
        let mut fire_if_allowed = |body: &serde_json::Value| -> Result<(), EndpointAdapterError> {
            gate_chat_completions_tools(body)?;
            mutations.fetch_add(1, Ordering::SeqCst);
            Ok(())
        };

        let with_tools = serde_json::json!({
            "model": "m",
            "tools": [{ "type": "function", "function": { "name": "x" } }],
            "messages": []
        });

        assert_eq!(
            endpoint_adapter_status(EndpointAdapterKind::OpenAiChatCompletions),
            EndpointAdapterStatus::AvailableLossless
        );
        fire_if_allowed(&with_tools).expect("lossless adapter allows tools");
        assert_eq!(mutations.load(Ordering::SeqCst), 1);

        let before = mutations.load(Ordering::SeqCst);
        let deny_unavailable = |body: &serde_json::Value| -> Result<(), EndpointAdapterError> {
            let _ = body;
            Err(EndpointAdapterError::Unavailable {
                endpoint: "/v1/chat/completions",
            })
        };
        let deny_lossy = |body: &serde_json::Value| -> Result<(), EndpointAdapterError> {
            let _ = body;
            Err(EndpointAdapterError::Lossy {
                endpoint: "/v1/chat/completions",
            })
        };

        let run_gated =
            |gate: &dyn Fn(&serde_json::Value) -> Result<(), EndpointAdapterError>| match gate(
                &with_tools,
            ) {
                Ok(()) => {
                    mutations.fetch_add(1, Ordering::SeqCst);
                }
                Err(_) => {}
            };

        run_gated(&deny_unavailable);
        assert_eq!(
            mutations.load(Ordering::SeqCst),
            before,
            "unavailable adapter must not fire mutation counter"
        );
        let msg = EndpointAdapterError::Unavailable {
            endpoint: "/v1/chat/completions",
        }
        .to_string();
        assert!(msg.contains("unavailable"), "{msg}");

        run_gated(&deny_lossy);
        assert_eq!(
            mutations.load(Ordering::SeqCst),
            before,
            "lossy adapter must not fire mutation counter"
        );
        let msg = EndpointAdapterError::Lossy {
            endpoint: "/v1/chat/completions",
        }
        .to_string();
        assert!(msg.contains("lossy"), "{msg}");

        assert!(gate_chat_completions_tools(&with_tools).is_ok());
    }

    #[test]
    fn tools_absent_bypasses_adapter_capability_gate() {
        let mutations = std::sync::atomic::AtomicUsize::new(0);
        let bodies = [
            serde_json::json!({ "model": "m", "messages": [] }),
            serde_json::json!({ "model": "m", "tools": [], "messages": [] }),
            serde_json::json!({ "model": "m", "tools": null, "messages": [] }),
        ];
        for body in &bodies {
            gate_chat_completions_tools(body).expect("tool-free must bypass adapter capability");
            mutations.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        assert_eq!(
            mutations.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "absent/empty tools must not block the mutation/dispatch path"
        );
    }

    #[test]
    fn endpoint_adapter_registry_covers_all_declared_kinds() {
        const ALL_KINDS: &[EndpointAdapterKind] = &[EndpointAdapterKind::OpenAiChatCompletions];

        for kind in ALL_KINDS {
            let status = endpoint_adapter_status(*kind);
            match kind {
                EndpointAdapterKind::OpenAiChatCompletions => {
                    assert_eq!(status, EndpointAdapterStatus::AvailableLossless);
                }
            }
            assert_eq!(status, EndpointAdapterRegistry::status(*kind));
        }

        assert!(
            ALL_KINDS
                .iter()
                .any(|k| endpoint_adapter_status(*k) == EndpointAdapterStatus::AvailableLossless),
            "registry must declare at least one AvailableLossless adapter"
        );
    }

}
