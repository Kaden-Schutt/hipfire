// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The generic autoregressive generate path.
//!
//! `generate` is the fallback every model without a specialised route takes,
//! and it was the last large body of daemon code manipulating architecture
//! types directly. It moves here with its transitive closure — the
//! `GenerationRoute` selector, the `qwen_ar_*` commit/terminal/routing
//! helpers, `QwenArSemanticProducer`, and the checkpoint knobs — because
//! those are reachable from nothing else, and splitting them would strand a
//! helper tail shared across crates. That shared tail is exactly what made
//! three earlier parallel attempts at this move fail.
//!
//! Moved verbatim: identical branch order, identical GPU dispatch, identical
//! wire strings and `unwrap_or` defaults.

use crate::common::*;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative;
use hipfire_engine::emit::*;
use hipfire_engine::redline::*;
use hipfire_engine::terminal::*;
use hipfire_loader::LoadedModel;
use hipfire_runtime::emit_text::{
    currently_in_think, ThinkOutputRouter, ToolOutputRouter, ToolRouteError,
};
use hipfire_runtime::emit_text::{ThinkRouteEvent, ToolRouteEvent};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::llama;
use hipfire_runtime::prompt_frame::ThinkMode;
use hipfire_runtime::sampler::{self, SamplerConfig};
use std::any::Any;
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "serve-fault-inject")]
thread_local! {
    static FAULT_AFTER_PREFILL_ARMED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(feature = "serve-fault-inject")]
pub fn arm_fault_after_prefill(armed: bool) {
    FAULT_AFTER_PREFILL_ARMED.with(|c| c.set(armed));
}

/// Route already-classified think/content events in order. Reasoning bypasses
/// tool parsing; answer content remains subject to ToolOutputRouter.
pub fn qwen_ar_route_think_events(
    stdout: &mut impl std::io::Write,
    id: &str,
    router: &mut ToolOutputRouter,
    channel_events: Vec<ThinkRouteEvent>,
    visible_acc: &mut String,
) -> Result<(), ToolRouteError> {
    for channel_event in channel_events {
        match channel_event {
            ThinkRouteEvent::Reasoning(reasoning) => {
                emit_reasoning_token(stdout, id, &reasoning);
            }
            ThinkRouteEvent::Content(content) => {
                let events = router.push(&content)?;
                for ev in events {
                    match ev {
                        ToolRouteEvent::VisibleText(vt) => {
                            visible_acc.push_str(vt.as_str());
                            emit_visible_token(stdout, id, vt.as_str());
                        }
                        ToolRouteEvent::ToolCall(_) => {
                            // Retained in router.tool_calls(); safe terminal releases it.
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Feed EosFilter-emitted UTF-8 through think-channel routing, then pass only
/// answer content to the tool router.
pub fn qwen_ar_route_filter_text(
    stdout: &mut impl std::io::Write,
    id: &str,
    think_router: &mut ThinkOutputRouter,
    router: &mut ToolOutputRouter,
    text: &str,
    visible_acc: &mut String,
) -> Result<(), ToolRouteError> {
    let mut channel_events = Vec::new();
    think_router.push_into(text, &mut channel_events);
    qwen_ar_route_think_events(stdout, id, router, channel_events, visible_acc)
}

/// Outcome of finishing the Qwen AR semantic router for one turn.
#[derive(Debug)]
pub struct QwenArRouteFinish {
    /// Daemon `done.finish_reason`.
    pub finish_reason: &'static str,
    /// Calls to put on the wire and into the asst-turn fingerprint.
    /// Empty when not tool-safe (length / stop without calls / suppressed).
    pub wire_tool_calls: Vec<hipfire_runtime::prompt_frame::ToolCall>,
    /// Whether `asst_turn_cache` may store this turn.
    pub store_cache: bool,
    /// Trailing visible prose flushed by `finish` (already appended to visible_acc
    /// by [`qwen_ar_finish_route`]); caller still emits token events for these.
    pub trailing_visible: Vec<String>,
    /// Explicit terminal cause (EOT beats length on the same budget token).
    pub cause: QwenArTerminalCause,
}

/// Explicit AR terminal cause. Decoded/filter EOT wins over
/// `generated == max_tokens` when both land on the same token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenArTerminalCause {
    /// Decoded stop marker (`<|im_end|>` / `<|endoftext|>`) via filter.
    DecodedEot,
    /// Hit the requested max_tokens budget without a decoded EOT.
    LengthCap,
    /// Natural stop (no length, no EOT marker — model finished cleanly).
    NaturalStop,
    /// Open think span at finish — fail-closed validation (no cache).
    OpenThink,
}

impl QwenArTerminalCause {
    /// Resolve cause: decoded EOT beats length on the same token.
    pub fn resolve(stopped_by_filter: bool, hit_length_cap: bool, open_think: bool) -> Self {
        if open_think {
            return Self::OpenThink;
        }
        if stopped_by_filter {
            return Self::DecodedEot;
        }
        if hit_length_cap {
            return Self::LengthCap;
        }
        Self::NaturalStop
    }
}

/// Finish the AR router. Length-cap (without EOT) never exposes tool_calls
/// or primes asst_turn_cache. Decoded EOT on the final budget token still
/// classifies as stop/tool_calls. Unclosed/malformed without length → Err.
/// Open think → non-retryable validation terminal, no cache, no hidden bytes.
pub fn qwen_ar_finish_route(
    router: ToolOutputRouter,
    cause: QwenArTerminalCause,
    visible_acc: &mut String,
) -> Result<QwenArRouteFinish, ToolRouteError> {
    if matches!(cause, QwenArTerminalCause::OpenThink) {
        return Ok(QwenArRouteFinish {
            finish_reason: "error",
            wire_tool_calls: Vec::new(),
            store_cache: false,
            trailing_visible: Vec::new(),
            cause,
        });
    }
    let length_unsafe = matches!(cause, QwenArTerminalCause::LengthCap);
    let buffered_before = router.tool_calls().to_vec();
    match router.finish() {
        Err(err) => {
            if length_unsafe {
                // Any pure-length terminal is unsafe: no calls, no cache.
                Ok(QwenArRouteFinish {
                    finish_reason: "length",
                    wire_tool_calls: Vec::new(),
                    store_cache: false,
                    trailing_visible: Vec::new(),
                    cause,
                })
            } else {
                Err(err)
            }
        }
        Ok(events) => {
            let mut trailing_visible = Vec::new();
            let mut finished_calls = buffered_before;
            for ev in events {
                match ev {
                    ToolRouteEvent::VisibleText(vt) => {
                        visible_acc.push_str(vt.as_str());
                        trailing_visible.push(vt.into_string());
                    }
                    ToolRouteEvent::ToolCall(tc) => {
                        finished_calls.push(tc);
                    }
                }
            }
            if length_unsafe {
                // Length is never a tool-safe or cache-safe terminal —
                // prose-only, complete hidden call, or trailing flush alike.
                Ok(QwenArRouteFinish {
                    finish_reason: "length",
                    wire_tool_calls: Vec::new(),
                    store_cache: false,
                    trailing_visible,
                    cause,
                })
            } else if !finished_calls.is_empty() {
                Ok(QwenArRouteFinish {
                    finish_reason: "tool_calls",
                    wire_tool_calls: finished_calls,
                    store_cache: true,
                    trailing_visible,
                    cause,
                })
            } else {
                Ok(QwenArRouteFinish {
                    finish_reason: "stop",
                    wire_tool_calls: Vec::new(),
                    store_cache: true,
                    trailing_visible,
                    cause,
                })
            }
        }
    }
}

/// EosFilter config for Qwen AR contract-v2 producer path. EosFilter owns
/// UTF-8/EOT filtering only; ThinkOutputRouter owns think-channel routing.
pub fn qwen_ar_eos_filter_config() -> EosFilterConfig {
    EosFilterConfig {
        strip_think: false,
        started_in_think: false,
        stop_at: vec![b"<|im_end|>".to_vec(), b"<|endoftext|>".to_vec()],
        holdback_prefixes: Vec::new(),
    }
}

/// Apply one filter observe step into the semantic router. Returns
/// `Ok(true)` when the filter signals Stop / EmitAndStop (decoded EOT)
/// so the caller can break the decode loop without emitting marker text.
pub fn qwen_ar_observe_and_route(
    stdout: &mut impl std::io::Write,
    id: &str,
    filter: &mut EosFilter,
    think_router: &mut ThinkOutputRouter,
    router: &mut ToolOutputRouter,
    new_bytes: &[u8],
    visible_acc: &mut String,
) -> Result<bool, ToolRouteError> {
    match filter.observe(new_bytes) {
        FilterAction::Emit(text_bytes) => {
            let text = std::str::from_utf8(&text_bytes).unwrap_or("");
            if !text.is_empty() {
                qwen_ar_route_filter_text(stdout, id, think_router, router, text, visible_acc)?;
            }
            Ok(false)
        }
        FilterAction::EmitAndStop(text_bytes) => {
            let text = std::str::from_utf8(&text_bytes).unwrap_or("");
            if !text.is_empty() {
                qwen_ar_route_filter_text(stdout, id, think_router, router, text, visible_acc)?;
            }
            Ok(true)
        }
        FilterAction::Hold => Ok(false),
        FilterAction::Stop => Ok(true),
    }
}

/// Shared end-of-stream drain used by production decode and deterministic
/// tests. Flushes EOT-prefix prose through think routing, then classifies any
/// trailing partial think marker as ordinary text in its current channel.
pub fn qwen_ar_drain_pending_into_router(
    stdout: &mut impl std::io::Write,
    id: &str,
    filter: &mut EosFilter,
    think_router: &mut ThinkOutputRouter,
    router: &mut ToolOutputRouter,
    visible_acc: &mut String,
) -> Result<(), ToolRouteError> {
    let pending = filter.flush_pending();
    if !pending.is_empty() {
        let text = std::str::from_utf8(&pending).unwrap_or("");
        if !text.is_empty() {
            qwen_ar_route_filter_text(stdout, id, think_router, router, text, visible_acc)?;
        }
    }

    let mut channel_events = Vec::new();
    think_router.finish_into(&mut channel_events);
    qwen_ar_route_think_events(stdout, id, router, channel_events, visible_acc)
}

/// Raw-commit bookkeeping shared by production and tests. Advances
/// conversation/stream/seq_pos together before any fallible classify step.
/// Returns the stream position of the newly committed token.
/// Whether a raw-committed token is classified-visible (goes through the
/// client decode/filter path) or intentionally hidden (state-only, e.g. the
/// post-EOT ChatML `\n` trailer that must never become client-visible prose).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenArRawCommitDisposition {
    /// Token participates in streamed decode / classify.
    ClassifiedVisible,
    /// Token mutates conversation/KV bookkeeping only; not streamed.
    IntentionallyHidden,
}

/// Sole producer-owned raw commit for every state-mutating Qwen AR token.
///
/// `ClassifiedVisible` pushes conversation + streamed and advances `seq_pos`.
/// `IntentionallyHidden` pushes conversation only (no streamed bytes) and
/// advances `seq_pos` so trailer tokens stay client-invisible while still
/// going through the same bookkeeping entry as visible tokens.
pub fn qwen_ar_raw_commit_token(
    conversation_tokens: &mut Vec<u32>,
    streamed_tokens: &mut Vec<u32>,
    seq_pos: &mut usize,
    token: u32,
    disposition: QwenArRawCommitDisposition,
) -> usize {
    conversation_tokens.push(token);
    match disposition {
        QwenArRawCommitDisposition::ClassifiedVisible => {
            streamed_tokens.push(token);
            *seq_pos += 1;
            streamed_tokens.len() - 1
        }
        QwenArRawCommitDisposition::IntentionallyHidden => {
            // Hidden tokens still advance physical seq_pos (KV write already
            // happened or will); they do not join the client stream index.
            *seq_pos += 1;
            // Return conversation index so callers can assert exactly-once
            // mutation without inventing a streamed position.
            conversation_tokens.len() - 1
        }
    }
}

/// Cache-store action derived from a finish. Production and tests share this
/// so `store_cache` is never asserted in isolation from the sink mutation.
#[derive(Debug, Clone)]
pub struct QwenArCacheAction {
    pub store: bool,
    pub fingerprint_text: String,
    pub tool_calls: Vec<hipfire_runtime::prompt_frame::ToolCall>,
}

pub fn qwen_ar_cache_action(
    finish: &QwenArRouteFinish,
    visible_for_cache: &str,
) -> QwenArCacheAction {
    QwenArCacheAction {
        store: finish.store_cache,
        fingerprint_text: crate::common::normalize_asst_turn_for_fingerprint(visible_for_cache),
        tool_calls: finish.wire_tool_calls.clone(),
    }
}

/// Apply a cache action through a shared insert seam (production `AsstTurnCache`
/// or a test `HashMap`). Returns the fingerprint key when a store happened.
pub fn qwen_ar_apply_cache_action<F>(
    mut insert: F,
    action: &QwenArCacheAction,
    cached_seq: Vec<u32>,
) -> Option<u64>
where
    F: FnMut(u64, Vec<u32>),
{
    if !action.store || cached_seq.is_empty() {
        return None;
    }
    let fp = crate::common::asst_turn_fingerprint(&action.fingerprint_text, &action.tool_calls);
    insert(fp, cached_seq);
    Some(fp)
}

/// Build the full Qwen AR `done` envelope (hostile-id safe). Used to stage
/// `commit_ready` then emit identical payload as `done` after Commit.
pub fn qwen_ar_done_value(
    id: &str,
    finish_reason: &str,
    generated: usize,
    tok_s: f64,
    prefill_tokens: usize,
    prefill_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    ttft_ms: f64,
    cached_tokens: usize,
    pflash_fragment_json: &str,
) -> serde_json::Value {
    let mut envelope = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": (tok_s * 10.0).round() / 10.0,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": (prefill_ms * 10.0).round() / 10.0,
        "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
        "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
        "ttft_ms": (ttft_ms * 10.0).round() / 10.0,
        "cached_tokens": cached_tokens,
        "finish_reason": finish_reason,
        "attempt_id": active_attempt_id(),
    });
    // Optional pflash object already serialized as `, "pflash":{...}` fragment
    // by production; parse and merge when non-empty so the writer stays serde.
    if !pflash_fragment_json.is_empty() {
        let padded = format!("{{{}}}", pflash_fragment_json.trim_start_matches(','));
        if let Ok(serde_json::Value::Object(map)) =
            serde_json::from_str::<serde_json::Value>(&padded)
        {
            for (k, v) in map {
                envelope[k] = v;
            }
        }
    }
    envelope
}

/// Emit open-think fail-closed validation terminal: exactly one correlated
/// non-retryable validation `error` and **no** `done` (terminal XOR).
/// Uses the production fail-closed error writer; `rolled_back` must come from
/// a completed [`crate::common::RollbackEpilogue`] (or false when no GPU reset ran).
pub fn emit_qwen_ar_open_think_terminal(
    stdout: &mut impl std::io::Write,
    id: &str,
    _generated: usize,
    epilogue: &crate::common::RollbackEpilogue,
) {
    crate::common::emit_fail_closed_error(
        stdout,
        Some(id),
        "open think span at end of generation (validation)",
        "validation",
        false,
        epilogue,
    );
}

/// Deterministic Qwen AR semantic producer used by production finish
/// orchestration and unit tests. Owns filter + router state and emits only
/// through the same helpers the GPU path uses.
pub struct QwenArSemanticProducer {
    pub id: String,
    pub filter: EosFilter,
    pub think_router: ThinkOutputRouter,
    pub router: ToolOutputRouter,
    pub visible_acc: String,
    /// Tokens that completed raw-commit before classify for this producer.
    pub raw_committed: Vec<u32>,
    /// Stream positions recorded at each raw commit (testable ordering).
    pub raw_commit_positions: Vec<usize>,
    pub stopped_by_filter: bool,
}

impl QwenArSemanticProducer {
    pub fn new(id: impl Into<String>, started_in_think: bool) -> Self {
        Self::new_with_tool_protocol(id, started_in_think, true)
    }

    pub fn new_with_tool_protocol(
        id: impl Into<String>,
        started_in_think: bool,
        tool_protocol_enabled: bool,
    ) -> Self {
        Self {
            id: id.into(),
            filter: EosFilter::new(qwen_ar_eos_filter_config()),
            think_router: ThinkOutputRouter::new(started_in_think),
            router: if tool_protocol_enabled {
                ToolOutputRouter::new()
            } else {
                ToolOutputRouter::disabled()
            },
            visible_acc: String::new(),
            raw_committed: Vec::new(),
            raw_commit_positions: Vec::new(),
            stopped_by_filter: false,
        }
    }

    /// Sole production/test raw-commit entry parameterized by disposition.
    ///
    /// `raw_commit` runs first and must return `(pos, new_bytes)`. It must not
    /// borrow the wire writer (avoids aliasing with classify). Producer
    /// bookkeeping (`raw_committed` / `raw_commit_positions`) is stamped
    /// atomically here and nowhere else.
    ///
    /// - [`ClassifiedVisible`](QwenArRawCommitDisposition::ClassifiedVisible):
    ///   `on_committed(pos, stdout)` then filter → router classify.
    /// - [`IntentionallyHidden`](QwenArRawCommitDisposition::IntentionallyHidden):
    ///   stamps only — no wire emit, no classify (ChatML trailer, etc.).
    pub fn commit_raw<F, B, E>(
        &mut self,
        stdout: &mut impl std::io::Write,
        token: u32,
        disposition: QwenArRawCommitDisposition,
        mut raw_commit: F,
        mut on_committed: E,
    ) -> Result<bool, ToolRouteError>
    where
        F: FnMut() -> (usize, B),
        B: AsRef<[u8]>,
        E: FnMut(usize, &mut dyn std::io::Write),
    {
        let (pos, new_bytes) = raw_commit();
        self.raw_committed.push(token);
        self.raw_commit_positions.push(pos);
        match disposition {
            QwenArRawCommitDisposition::ClassifiedVisible => {
                on_committed(pos, stdout);
                if self.stopped_by_filter {
                    return Ok(true);
                }
                let stop = qwen_ar_observe_and_route(
                    stdout,
                    &self.id,
                    &mut self.filter,
                    &mut self.think_router,
                    &mut self.router,
                    new_bytes.as_ref(),
                    &mut self.visible_acc,
                )?;
                if stop {
                    self.stopped_by_filter = true;
                }
                Ok(stop)
            }
            QwenArRawCommitDisposition::IntentionallyHidden => {
                // Hidden: physical/state mutation only; never classify or emit.
                let _ = (new_bytes, &mut on_committed);
                Ok(self.stopped_by_filter)
            }
        }
    }

    /// Classified-visible commit-then-classify (production decode / inject).
    ///
    /// Thin wrapper over [`Self::commit_raw`] with
    /// [`ClassifiedVisible`](QwenArRawCommitDisposition::ClassifiedVisible).
    pub fn commit_and_classify<F, B, E>(
        &mut self,
        stdout: &mut impl std::io::Write,
        token: u32,
        raw_commit: F,
        on_committed: E,
    ) -> Result<bool, ToolRouteError>
    where
        F: FnMut() -> (usize, B),
        B: AsRef<[u8]>,
        E: FnMut(usize, &mut dyn std::io::Write),
    {
        self.commit_raw(
            stdout,
            token,
            QwenArRawCommitDisposition::ClassifiedVisible,
            raw_commit,
            on_committed,
        )
    }

    /// Convenience: raw-commit conversation/stream/seq_pos then classify.
    pub fn commit_and_observe(
        &mut self,
        stdout: &mut impl std::io::Write,
        conversation_tokens: &mut Vec<u32>,
        streamed_tokens: &mut Vec<u32>,
        seq_pos: &mut usize,
        token: u32,
        new_bytes: &[u8],
    ) -> Result<bool, ToolRouteError> {
        let owned = new_bytes.to_vec();
        self.commit_raw(
            stdout,
            token,
            QwenArRawCommitDisposition::ClassifiedVisible,
            || {
                let pos = qwen_ar_raw_commit_token(
                    conversation_tokens,
                    streamed_tokens,
                    seq_pos,
                    token,
                    QwenArRawCommitDisposition::ClassifiedVisible,
                );
                (pos, owned.clone())
            },
            |_pos, _out| {},
        )
    }

    /// Shared finish: drain pending, finalize think routing, then classify.
    /// Emits trailing visible prose only. Tool-call release is deferred to the
    /// caller until after a successful client terminal Commit decision.
    /// Returns `(route_finish, final_visible_text)`.
    pub fn finish(
        mut self,
        stdout: &mut impl std::io::Write,
        hit_length_cap: bool,
    ) -> Result<(QwenArRouteFinish, String), ToolRouteError> {
        qwen_ar_drain_pending_into_router(
            stdout,
            &self.id,
            &mut self.filter,
            &mut self.think_router,
            &mut self.router,
            &mut self.visible_acc,
        )?;
        let open_think = self.think_router.in_think();
        let cause =
            QwenArTerminalCause::resolve(self.stopped_by_filter, hit_length_cap, open_think);
        if matches!(cause, QwenArTerminalCause::OpenThink) {
            // Fail-closed: no calls/cache/done. Caller owns the
            // production rollback epilogue + single correlated error terminal
            // (tests call `emit_qwen_ar_open_think_terminal` after finish).
            let finish = QwenArRouteFinish {
                finish_reason: "error",
                wire_tool_calls: Vec::new(),
                store_cache: false,
                trailing_visible: Vec::new(),
                cause,
            };
            return Ok((finish, String::new()));
        }
        let finish = qwen_ar_finish_route(self.router, cause, &mut self.visible_acc)?;
        for trailing in &finish.trailing_visible {
            emit_visible_token(stdout, &self.id, trailing);
        }
        // Tool-call events stay deferred until Commit (see
        // `crate::qwen::qwen_client_commit_effects` / production finish orchestration).
        Ok((finish, self.visible_acc))
    }

    pub fn visible(&self) -> &str {
        &self.visible_acc
    }
}

#[cfg(feature = "serve-fault-inject")]
pub fn take_fault_after_prefill() -> bool {
    FAULT_AFTER_PREFILL_ARMED.with(|c| {
        let v = c.get();
        c.set(false);
        v
    })
}

/// Emit a single-line `{"type":"error","id":"...","message":"..."}` JSON
/// line on the IPC stream. Uses `serde_json` so user-controlled error
/// strings (image decoder messages, base64 errors) can't desync the
/// protocol by injecting embedded `"`, `\`, or newline bytes.
pub fn write_error(stdout: &mut impl std::io::Write, id: &str, message: &str) {
    // Active-attempt internal error (echoes TLS attempt_id).
    crate::dense::emit_active_attempt_error(stdout, Some(id), message, "internal", false, false);
}

/// Fail-closed policy for ordinary single-GPU Qwen35 AR when
/// `forward_prefill_batch` / `forward_scratch` returns `Err` (VMM map/growth,
/// allocation, or other HipResult failure). Injected map failure must never
/// panic the daemon or emit a token whose trunk KV write did not commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenArForwardFailAction {
    /// Emit one request-scoped `{"type":"error",...}` and stop the request.
    pub emit_request_error: bool,
    /// Cold-reset uncommitted DN/KV/seq so unload + retry stay safe.
    pub reset_uncommitted_state: bool,
    /// Must stay false: never emit/commit the token whose KV write failed.
    pub emit_failed_token: bool,
    /// Adaptive poison is sticky across request failures.
    pub clear_adaptive_poison: bool,
}

pub fn qwen_ar_forward_fail_action() -> QwenArForwardFailAction {
    QwenArForwardFailAction {
        emit_request_error: true,
        reset_uncommitted_state: true,
        emit_failed_token: false,
        clear_adaptive_poison: false,
    }
}

pub fn qwen_ar_forward_fail_message(phase: &str, err: impl std::fmt::Display) -> String {
    format!("{phase}: {err}")
}

/// Bound an eviction-enabled Qwen prefill write while an adaptive cache still
/// owns the layout.  Before the one-way handoff, the eviction window is not a
/// capacity limit (its gate is closed); writes must instead stop at every
/// adaptive transition boundary.  After handoff, the normal budget window
/// again determines how much can be appended before compaction.
pub fn qwen_ar_eviction_prefill_chunk_limit(
    seq_pos: usize,
    eviction_window: usize,
    adaptive_staging: bool,
) -> usize {
    if adaptive_staging {
        qwen35::PREFILL_MAX_BATCH
    } else {
        eviction_window.saturating_sub(seq_pos).max(1)
    }
}

pub fn ckpt_resume_enabled() -> bool {
    std::env::var("HIPFIRE_CACHE_CKPT_RESUME").ok().as_deref() != Some("0")
}
pub fn ckpt_interval() -> usize {
    std::env::var("HIPFIRE_CACHE_CKPT_INTERVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048)
        .max(256)
}
pub fn ckpt_max() -> usize {
    std::env::var("HIPFIRE_CACHE_CKPT_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .max(1)
}

/// Truncate a checkpoint ring to `keep` slots, freeing the dropped snapshots'
/// GPU buffers (a bare `Vec::truncate` would leak them).
pub fn truncate_checkpoints(
    cks: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    keep: usize,
    gpu: &mut rdna_compute::Gpu,
) {
    while cks.len() > keep {
        if let Some((_, snap)) = cks.pop() {
            snap.free_gpu(gpu);
        }
    }
}

/// Exhaustive producer-route identity for one generate turn.
///
/// Selected once at the top of [`generate`] and is the sole authority for
/// dispatch branch choice and tools capability. Precedence matches production:
/// EP → arch short-circuits (Qwen2, DeepSeek4, LFM, Cohere, MiniMax, dots) →
/// pp>1 → Qwen/LLaMA DFlash/spec (MTP uses the generic wrapper) → default AR/unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationRoute {
    QwenAr,
    QwenDflash,
    Qwen2Ar,
    Qwen2Spec,
    Deepseek4Ar,
    Deepseek4Ep,
    Deepseek4Spec,
    CohereAr,
    CohereSpec,
    MiniMaxAr,
    MiniMaxEp,
    MiniMaxSpec,
    LfmAr,
    LfmSpec,
    LlamaAr,
    LlamaSpec,
    GlimmerAr,
    GlimmerSpec,
    PipelineParallel,
    /// Device-mesh dense carrier (Tp axis `tp_model` / dense Pp axis
    /// `pp_model`): weights + KV live inside the carrier; the in-process
    /// dense routes cannot serve them.
    MeshCarrier,
    DotsOcr,
    Unknown,
}

impl GenerationRoute {
    /// Every variant — coverage guard for table-driven tests.
    #[allow(dead_code)]
    pub const ALL: &'static [Self] = &[
        Self::QwenAr,
        Self::QwenDflash,
        Self::Qwen2Ar,
        Self::Qwen2Spec,
        Self::Deepseek4Ar,
        Self::Deepseek4Ep,
        Self::Deepseek4Spec,
        Self::CohereAr,
        Self::CohereSpec,
        Self::MiniMaxAr,
        Self::MiniMaxEp,
        Self::MiniMaxSpec,
        Self::LfmAr,
        Self::LfmSpec,
        Self::LlamaAr,
        Self::LlamaSpec,
        Self::GlimmerAr,
        Self::GlimmerSpec,
        Self::PipelineParallel,
        Self::MeshCarrier,
        Self::DotsOcr,
        Self::Unknown,
    ];

    /// Proven semantic-safe producers for non-empty tools.
    /// Exactly: Qwen AR, Qwen DFlash/spec, DS4 AR, DS4 EP, DS4 spec, Glimmer AR, Glimmer spec.
    pub const fn supports_tools(self) -> bool {
        matches!(
            self,
            Self::QwenAr
                | Self::QwenDflash
                | Self::Deepseek4Ar
                | Self::Deepseek4Ep
                | Self::Deepseek4Spec
                | Self::GlimmerAr
                | Self::GlimmerSpec
        )
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::QwenAr => "qwen_ar",
            Self::QwenDflash => "qwen_dflash",
            Self::Qwen2Ar => "qwen2_ar",
            Self::Qwen2Spec => "qwen2_spec",
            Self::Deepseek4Ar => "deepseek4_ar",
            Self::Deepseek4Ep => "deepseek4_ep",
            Self::Deepseek4Spec => "deepseek4_spec",
            Self::CohereAr => "cohere_ar",
            Self::CohereSpec => "cohere_spec",
            Self::MiniMaxAr => "minimax_ar",
            Self::MiniMaxEp => "minimax_ep",
            Self::MiniMaxSpec => "minimax_spec",
            Self::LfmAr => "lfm_ar",
            Self::LfmSpec => "lfm_spec",
            Self::LlamaAr => "llama_ar",
            Self::LlamaSpec => "llama_spec",
            Self::GlimmerAr => "glimmer_ar",
            Self::GlimmerSpec => "glimmer_spec",
            Self::PipelineParallel => "pipeline_parallel",
            Self::MeshCarrier => "mesh_carrier",
            Self::DotsOcr => "dots_ocr",
            Self::Unknown => "unknown",
        }
    }
}

/// Pure inputs for [`select_generation_route`]. No GPU/env side effects.
#[derive(Debug, Clone, Copy)]
pub struct GenerationRouteInputs {
    pub arch_id: u32,
    pub ep: bool,
    pub pp: usize,
    pub has_speculator: bool,
    pub speculator_is_mtp: bool,
    pub deepseek4_spec_requested: bool,
    pub ngram_can_sample: bool,
    pub temp: f32,
    pub user_explicit_sampling: bool,
    pub min_p: Option<f32>,
    /// At least one repeat/presence/frequency penalty is non-neutral. Sampled
    /// DFlash chain verify does not implement these controls and must use AR.
    pub nonneutral_penalties: bool,
    pub force_ar_chat: bool,
    pub temp_spec_env_off: bool,
    pub fast_sample_on: bool,
    pub supports_temp_swor: bool,
    /// Speculator applies faithful top_p/top_k on the sampled chain path
    /// (DFlash2 candidate-selector). When set, chain routing may engage even
    /// though [`Self::supports_temp_swor`] is also true; DDTree SWOR leaves
    /// this false and still refuses user-explicit non-temperature controls.
    pub supports_chain_nucleus_verify: bool,
    pub kv_adaptive: bool,
    /// A device-mesh dense carrier (admitted TP or dense PP) is loaded —
    /// weights/KV live inside the carrier, so no in-process route can serve it.
    pub mesh_carrier: bool,
}

/// Pure production route selector. Precedence is intentional and exhaustive.
pub fn select_generation_route(i: &GenerationRouteInputs) -> GenerationRoute {
    // 0. Device-mesh dense carriers first: their weights/KV live inside the
    // carrier, so no in-process route can serve them.
    if i.mesh_carrier {
        return GenerationRoute::MeshCarrier;
    }

    // 1. Expert-parallel first (before any arch short-circuit).
    if i.ep {
        return match i.arch_id {
            9 => GenerationRoute::Deepseek4Ep,
            10 => GenerationRoute::MiniMaxEp,
            // EP on an unregistered arch — still EP-served, not tool-safe.
            _ => GenerationRoute::Unknown,
        };
    }

    // 2. Arch short-circuits (Qwen2, DeepSeek4, LFM, Cohere, MiniMax, dots).
    match i.arch_id {
        7 => {
            let spec_ok = i.has_speculator && (i.temp <= 1e-6 || i.ngram_can_sample);
            return if spec_ok {
                GenerationRoute::Qwen2Spec
            } else {
                GenerationRoute::Qwen2Ar
            };
        }
        9 => {
            let spec_temp_ok = i.temp <= 1e-6 || i.ngram_can_sample;
            // deepseek4: ngram_can_sample mirrors !requires_greedy for ds4 drafters.
            let spec_mode = i.deepseek4_spec_requested && spec_temp_ok && i.has_speculator;
            return if spec_mode {
                GenerationRoute::Deepseek4Spec
            } else {
                GenerationRoute::Deepseek4Ar
            };
        }
        11 => {
            let spec_ok = i.has_speculator && (i.temp <= 1e-6 || i.ngram_can_sample);
            return if spec_ok {
                GenerationRoute::LfmSpec
            } else {
                GenerationRoute::LfmAr
            };
        }
        12 => {
            let spec_ok = i.has_speculator && (i.temp <= 1e-6 || i.ngram_can_sample);
            return if spec_ok {
                GenerationRoute::CohereSpec
            } else {
                GenerationRoute::CohereAr
            };
        }
        10 => {
            let spec_ok = i.has_speculator && (i.temp <= 1e-6 || i.ngram_can_sample);
            return if spec_ok {
                GenerationRoute::MiniMaxSpec
            } else {
                GenerationRoute::MiniMaxAr
            };
        }
        14 => {
            let spec_ok = i.has_speculator && (i.temp <= 1e-6 || i.ngram_can_sample);
            return if spec_ok {
                GenerationRoute::GlimmerSpec
            } else {
                GenerationRoute::GlimmerAr
            };
        }
        8 => return GenerationRoute::DotsOcr,
        _ => {}
    }

    // 3. Pipeline-parallel.
    if i.pp > 1 {
        return GenerationRoute::PipelineParallel;
    }

    // 4. Qwen / LLaMA DFlash/spec (same gates as production generate body).
    // MTP (speculator.name()=="mtp") shares the generic QwenDflash wrapper.
    // Sampled MTP honors temp>0 including user-explicit sampling and min_p
    // when supports_temp_verify; DFlash keeps SWOR vs selector-chain gates.
    // Selector-chain nucleus may share supports_temp_swor with DDTree SWOR but
    // still takes the sampled chain route (user-explicit top_p/top_k allowed;
    // min_p>0 still blocked — DFlash ignores min_p).
    let caps = hipfire_loader::carrier_for(i.arch_id)
        .map(|c| c.caps())
        .unwrap_or_default();
    let dflash_min_p_present = i.min_p.map(|p| p > 0.0).unwrap_or(false);
    let ddtree_swor_route = !i.speculator_is_mtp
        && i.temp > 1e-6
        && i.supports_temp_swor
        && !i.supports_chain_nucleus_verify
        && !i.user_explicit_sampling
        && !i.temp_spec_env_off;
    let chain_sample_route = !i.speculator_is_mtp
        && i.temp > 1e-6
        && (!i.supports_temp_swor || i.supports_chain_nucleus_verify)
        && i.ngram_can_sample
        && i.fast_sample_on
        && !dflash_min_p_present
        && !i.nonneutral_penalties
        && !i.temp_spec_env_off;
    let mtp_sample_route = i.speculator_is_mtp && i.temp > 1e-6 && i.supports_temp_swor;
    let qwen_dflash_route = caps.is_qwen_dflash()
        && (i.temp <= 1e-6 || ddtree_swor_route || chain_sample_route || mtp_sample_route);
    let llama_dflash_route =
        caps.is_llama_dflash() && (i.temp <= 1e-6 || ddtree_swor_route || chain_sample_route);
    if i.has_speculator
        && !i.force_ar_chat
        && (qwen_dflash_route || llama_dflash_route)
        && !(i.kv_adaptive && caps.spec_excludes_adaptive)
    {
        return if caps.is_llama_dflash() {
            GenerationRoute::LlamaSpec
        } else {
            GenerationRoute::QwenDflash
        };
    }

    // 5. Default AR / unknown.
    if caps.is_qwen_dflash() {
        GenerationRoute::QwenAr
    } else if caps.is_llama_dflash() {
        GenerationRoute::LlamaAr
    } else {
        GenerationRoute::Unknown
    }
}

#[inline]
pub fn llama_qwen3_batched_prefill_eligible(
    gpu_arch: &str,
    model_arch: llama::ModelArch,
    prefill_batched_enabled: bool,
    quant_q8: bool,
    has_eviction: bool,
    token_count: usize,
) -> bool {
    model_arch == llama::ModelArch::Qwen3
        && (gpu_arch.starts_with("gfx11") || gpu_arch == "gfx1201")
        && prefill_batched_enabled
        && quant_q8
        && !has_eviction
        && token_count >= 4
}

#[inline]
pub fn llama_prefill_sample_seed(mut seed: u32, token_count: usize, temperature: f32) -> u32 {
    // The legacy sequential prefill sampled after every prompt token. Sampling
    // at temperature=0 does not advance xorshift32; sampled generation advances
    // once per token. Batched prefill only samples the final logits, so advance
    // over the discarded intermediate draws to preserve the final draw + state.
    if temperature > 1e-6 {
        for _ in 1..token_count {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
        }
    }
    seed
}

#[allow(clippy::too_many_arguments)]
pub fn generate(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    drafter_gpu: Option<&mut rdna_compute::Gpu>,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    // Whether the request EXPLICITLY set a non-temperature sampling control
    // (top_p/top_k/min_p/penalties). Gates temp>0 spec routing: explicit controls
    // force the AR sampler (the SWOR spec verify can only honor temperature).
    user_explicit_sampling: bool,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: Option<f32>,
    // CACTUS acceptance-boost δ (0 = lossless). Request opt-in; applies only to a
    // CACTUS-capable sampled verify (deepseek4 DSpark / qwen35 DFlash).
    cactus_delta: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    pflash_state: Option<&mut hipfire_pflash::pflash::PflashState>,
    pflash_cfg: Option<&hipfire_pflash::pflash::PflashConfig>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    think_mode: ThinkMode,
    stop: &[String],
    reasoning_effort: Option<&str>,
    enable_thinking: bool,
    // `Some(k)` emits OpenAI logprobs with k candidates per token. `None` is the
    // default and leaves every token envelope byte-identical to before.
    logprobs_top_k: Option<usize>,
) {
    // ── Producer-route authority (Task 6) ──────────────────────────────
    // Resolve the selected generation route BEFORE sampler RNG reset and
    // every stateful branch. Non-empty tools are denied for unsafe routes
    // with one correlated non-retryable unsupported error (no gen_start/
    // done/calls/cache). Tool-free requests bypass the gate unchanged.
    let ngram_can_sample = m
        .speculator
        .as_ref()
        .map(|s| !s.requires_greedy())
        .unwrap_or(false);
    let supports_temp_swor = m
        .speculator
        .as_ref()
        .is_some_and(|s| s.supports_temp_verify());
    let supports_chain_nucleus_verify = m
        .speculator
        .as_ref()
        .is_some_and(|s| s.supports_chain_nucleus_verify());
    let route_inputs = GenerationRouteInputs {
        arch_id: m.arch_id,
        ep: m.ep.is_some(),
        pp: m.pp,
        has_speculator: m.speculator.is_some(),
        speculator_is_mtp: m.speculator.as_ref().is_some_and(|s| s.name() == "mtp"),
        deepseek4_spec_requested: deepseek4_spec_requested(m),
        ngram_can_sample,
        temp,
        user_explicit_sampling,
        min_p,
        nonneutral_penalties: repeat_penalty != 1.0
            || presence_penalty != 0.0
            || frequency_penalty != 0.0,
        force_ar_chat: std::env::var("HIPFIRE_DFLASH_CHAT").ok().as_deref() == Some("0"),
        temp_spec_env_off: std::env::var("HIPFIRE_DFLASH_TEMP_SPEC").ok().as_deref() == Some("0"),
        fast_sample_on: hipfire_runtime::config::get().dflash_fast_sample,
        supports_temp_swor,
        supports_chain_nucleus_verify,
        kv_adaptive: m.kv_adaptive.is_some(),
        mesh_carrier: m.tp_model.is_some() || m.pp_model.is_some(),
    };
    let selected_route = select_generation_route(&route_inputs);
    let tools_nonempty = tools.map(|t| !t.is_empty()).unwrap_or(false);
    if tools_nonempty && !selected_route.supports_tools() {
        crate::dense::emit_active_attempt_error(
            stdout,
            Some(id),
            &format!(
                "tools are not supported on producer route {} (semantic-safe producers: qwen_ar, qwen_dflash, deepseek4_ar, deepseek4_ep, deepseek4_spec)",
                selected_route.name()
            ),
            "unsupported",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    match hipfire_loader::generation_early_route(m.arch_id) {
        Some(hipfire_loader::GenerationEarlyRoute::Gemma4) => {
            // The loader publishes one of two mutually-exclusive Gemma4 states:
            // eager dense (ModelState::Gemma4) and lowered/MoE
            // (ModelState::Gemma4Lowered). The generate body is eager-only, so a
            // lowered load must fail loudly here rather than silently run eager
            // against lowered weights.
            if m.gemma4_lowered_mut().is_some() {
                emit_error_with_id(
                stdout,
                id,
                "gemma4 lowered/MoE generate not yet wired on this build (eager dense only) —                  reload without batched/WMMA prefill opt-in or the MoE variant",
            );
                return;
            }
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
                think_mode,
                user_explicit_sampling,
                top_k,
                min_p,
                cactus_delta,
            );
            let _ = (
                repeat_penalty,
                repeat_window,
                presence_penalty,
                frequency_penalty,
            );
            crate::dense::generate_gemma4(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                max_tokens,
                max_think_tokens,
                enable_thinking,
                tools,
                messages_history,
                stop,
                logprobs_top_k,
            );
            return;
        }
        Some(hipfire_loader::GenerationEarlyRoute::MuseGlimmer) => {
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
                user_explicit_sampling,
                cactus_delta,
            );
            let _ = (
                repeat_penalty,
                repeat_window,
                presence_penalty,
                frequency_penalty,
            );
            let _ = stop;
            crate::dense::generate_muse_glimmer(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                top_k.map(|k| k as usize).unwrap_or(0),
                min_p,
                max_tokens,
                max_think_tokens,
                think_mode,
                reasoning_effort,
                tools,
                messages_history,
            );
            return;
        }
        None => {}
    }

    // hunt3 M-E: seed the process-global CPU sampler RNG with this request's
    // fixed seed so the grammar/CPU-fallback sample stream is deterministic per
    // request and does not carry RNG state across requests. Matches the u32 the
    // GPU sample path uses (0x13579BDF).
    hipfire_runtime::llama::reset_cpu_sampler_rng(0x13579BDF);
    // Adaptive KV poison is sticky until unload/reload. Refuse generation so a
    // partial tier transition cannot continue writing into mixed-tier state.
    if let Some(ad) = m.kv_adaptive.as_ref() {
        if ad.is_poisoned() {
            let reason = ad
                .poison_reason()
                .unwrap_or("adaptive KV is poisoned; unload/reload required");
            write_error(stdout, id, reason);
            return;
        }
    }

    // Compress runs on the PFlash drafter handle when one is set (hetero
    // sibling device), else on the target gpu. The handle is consumed at
    // the seq_pos==0 compress site; decode always uses `gpu`.
    let mut drafter_gpu = drafter_gpu;

    // Dispatch is structurally authoritative on `selected_route`. Spec-capacity
    // miss fallthrough (crate::qwen::generate_dflash → false) stays inside the Spec arm and
    // continues to that arch's AR producer — never an independent re-predicate.
    match selected_route {
        GenerationRoute::Deepseek4Ep | GenerationRoute::MiniMaxEp => {
            // EP serve (ds4/minimax): thread the SAME resolved sampling the
            // single-GPU handler computed (request field > m.rec_* > arch-default
            // ladder, all done at the call site above) into the EP decode loops.
            // Previously the EP path dropped these to a hardcoded greedy argmax,
            // which loops on ds4's quantized instruct model (card mandates
            // temp=1.0/top_p=1.0). reset_cpu_sampler_rng(0x13579BDF) was already
            // called above, so the host-side draw in ep_serve_* is deterministic.
            let ep_sampling = crate::qwen::EpSampling {
                temp,
                top_p,
                top_k,
                min_p,
            };
            crate::qwen::generate_ep(
                m,
                stdout,
                id,
                prompt,
                system_prompt,
                max_tokens,
                max_think_tokens,
                think_mode,
                tools,
                messages_history,
                stop,
                ep_sampling,
            );
            return;
        }
        GenerationRoute::Unknown if m.ep.is_some() => {
            // EP on an unregistered arch_id — preserve tool-free EP serve.
            let ep_sampling = crate::qwen::EpSampling {
                temp,
                top_p,
                top_k,
                min_p,
            };
            crate::qwen::generate_ep(
                m,
                stdout,
                id,
                prompt,
                system_prompt,
                max_tokens,
                max_think_tokens,
                think_mode,
                tools,
                messages_history,
                stop,
                ep_sampling,
            );
            return;
        }
        GenerationRoute::Qwen2Spec => {
            if crate::qwen::generate_dflash(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                max_tokens,
                max_think_tokens,
                assistant_prefix,
                None, // pflash_bypass_reason — no pflash on the n-gram path
                None, // pflash_alpha
                tools,
                messages_history,
                stop,
                temp,
                top_p,
                top_k.map(|k| k as usize).unwrap_or(0),
                min_p.unwrap_or(0.0),
                cactus_delta,
                reasoning_effort,
                enable_thinking,
            ) {
                return;
            }
            // ctx-capacity miss: fall through to Qwen2 AR (same tokens, no spec).
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                max_think_tokens,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
                tools,
                messages_history,
            );
            let _ = stop;
            crate::dense::generate_qwen2(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                max_tokens,
                repeat_penalty,
                repeat_window,
            );
            return;
        }
        GenerationRoute::Qwen2Ar => {
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                max_think_tokens,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
                tools,
                messages_history,
            );
            let _ = stop;
            crate::dense::generate_qwen2(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                max_tokens,
                repeat_penalty,
                repeat_window,
            );
            return;
        }
        GenerationRoute::Deepseek4Spec => {
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                max_think_tokens,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
            );
            let _ = (repeat_penalty, repeat_window);
            let _ = stop;
            crate::dense::generate_deepseek4_spec(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                max_tokens,
                temp,
                top_p,
                top_k.map(|k| k as usize).unwrap_or(0),
                cactus_delta,
                think_mode,
                tools,
                messages_history,
            );
            return;
        }
        GenerationRoute::Deepseek4Ar => {
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                max_think_tokens,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
            );
            let _ = (repeat_penalty, repeat_window);
            let _ = stop;
            crate::dense::generate_deepseek4(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                max_tokens,
                think_mode,
                tools,
                messages_history,
            );
            return;
        }
        GenerationRoute::LfmSpec => {
            if crate::qwen::generate_dflash(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                max_tokens,
                max_think_tokens,
                assistant_prefix,
                None,
                None,
                tools,
                messages_history,
                stop,
                temp,
                top_p,
                top_k.map(|k| k as usize).unwrap_or(0),
                min_p.unwrap_or(0.0),
                cactus_delta,
                reasoning_effort,
                enable_thinking,
            ) {
                return;
            }
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
                think_mode,
            );
            let _ = (repeat_penalty, repeat_window);
            crate::dense::generate_lfm2moe(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                max_tokens,
                max_think_tokens,
                tools,
                messages_history,
            );
            return;
        }
        GenerationRoute::LfmAr => {
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
                think_mode,
            );
            let _ = (repeat_penalty, repeat_window);
            crate::dense::generate_lfm2moe(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                max_tokens,
                max_think_tokens,
                tools,
                messages_history,
            );
            return;
        }
        GenerationRoute::CohereSpec => {
            if crate::qwen::generate_dflash(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                max_tokens,
                max_think_tokens,
                assistant_prefix,
                None,
                None,
                tools,
                messages_history,
                stop,
                temp,
                top_p,
                top_k.map(|k| k as usize).unwrap_or(0),
                min_p.unwrap_or(0.0),
                cactus_delta,
                reasoning_effort,
                enable_thinking,
            ) {
                return;
            }
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
                think_mode,
            );
            let _ = (repeat_penalty, repeat_window);
            crate::dense::generate_cohere2moe(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                max_tokens,
                max_think_tokens,
                tools,
                messages_history,
            );
            return;
        }
        GenerationRoute::CohereAr => {
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
                think_mode,
            );
            let _ = (repeat_penalty, repeat_window);
            crate::dense::generate_cohere2moe(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                max_tokens,
                max_think_tokens,
                tools,
                messages_history,
            );
            return;
        }
        GenerationRoute::MiniMaxSpec => {
            if crate::qwen::generate_dflash(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                max_tokens,
                max_think_tokens,
                assistant_prefix,
                None,
                None,
                tools,
                messages_history,
                stop,
                temp,
                top_p,
                top_k.map(|k| k as usize).unwrap_or(0),
                min_p.unwrap_or(0.0),
                cactus_delta,
                reasoning_effort,
                enable_thinking,
            ) {
                return;
            }
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
                think_mode,
            );
            let _ = (repeat_penalty, repeat_window);
            crate::dense::generate_minimax(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                max_tokens,
                max_think_tokens,
                tools,
                messages_history,
            );
            return;
        }
        GenerationRoute::MiniMaxAr => {
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
                think_mode,
            );
            let _ = (repeat_penalty, repeat_window);
            crate::dense::generate_minimax(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                max_tokens,
                max_think_tokens,
                tools,
                messages_history,
            );
            return;
        }
        GenerationRoute::DotsOcr => {
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                max_think_tokens,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
                tools,
                messages_history,
            );
            let _ = (repeat_penalty, repeat_window);
            let _ = stop;
            let _ = (top_k, min_p, presence_penalty, frequency_penalty);
            crate::vision::generate_dots_ocr_text(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                max_tokens,
            );
            return;
        }
        GenerationRoute::PipelineParallel => {
            crate::qwen::generate_multi(
                m,
                gpu,
                pflash_state,
                pflash_cfg,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                top_k,
                min_p,
                max_tokens,
                repeat_penalty,
                repeat_window,
                presence_penalty,
                frequency_penalty,
                budget_alert_at_tok,
                budget_alert_text,
                max_think_tokens,
                assistant_prefix,
                tools,
                messages_history,
                stop,
                reasoning_effort,
                enable_thinking,
            );
            return;
        }
        GenerationRoute::MeshCarrier => {
            let _ = (
                budget_alert_at_tok,
                budget_alert_text,
                max_think_tokens,
                assistant_prefix,
                pflash_state,
                pflash_cfg,
                think_mode,
                reasoning_effort,
                enable_thinking,
            );
            crate::mesh::generate_mesh_carrier(
                m,
                stdout,
                id,
                prompt,
                system_prompt,
                temp,
                top_p,
                top_k,
                min_p,
                max_tokens,
                repeat_penalty,
                repeat_window,
                presence_penalty,
                frequency_penalty,
                tools,
                messages_history,
                stop,
            );
            return;
        }
        GenerationRoute::QwenDflash | GenerationRoute::LlamaSpec | GenerationRoute::GlimmerSpec => {
            // Operator visibility: a temp>0 request on a DFlash-capable arch that
            // did NOT qualify is handled by the selector (falls to AR). When we
            // are on the DFlash arm, still warn once if min_p was requested.
            let minp_requested = min_p.map(|p| p > 0.0).unwrap_or(false);
            let spec_is_mtp = m.speculator.as_ref().is_some_and(|s| s.name() == "mtp");
            if temp > 1e-6 && minp_requested && !spec_is_mtp {
                static SPEC_MINP_WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !SPEC_MINP_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"warning","id":"{}","message":"DFlash spec-decode honors temp+top_p+top_k but ignores min_p; set HIPFIRE_DFLASH_CHAT=0 to route through AR for full min_p support"}}"#,
                        id,
                    );
                    let _ = stdout.flush();
                }
            }
            let mut dflash_bypass_reason: Option<&'static str> = None;
            let dflash_alpha = pflash_cfg.as_ref().map(|c| c.alpha);
            if let Some(cfg) = pflash_cfg.as_ref() {
                if cfg.mode != hipfire_pflash::pflash::PflashMode::Off {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_bypass","id":"{}","reason":"dflash_decode_active (pflash compression on the DFlash path is a follow-up; set dflash_mode=off to compress with AR decode)"}}"#,
                        id,
                    );
                    let _ = stdout.flush();
                    dflash_bypass_reason = Some("dflash_decode_active");
                }
            }
            if crate::qwen::generate_dflash(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                max_tokens,
                max_think_tokens,
                assistant_prefix,
                dflash_bypass_reason,
                dflash_alpha,
                tools,
                messages_history,
                stop,
                temp,
                top_p,
                top_k.map(|k| k as usize).unwrap_or(0),
                min_p.unwrap_or(0.0),
                cactus_delta,
                reasoning_effort,
                enable_thinking,
            ) {
                let _ = (
                    repeat_penalty,
                    repeat_window,
                    budget_alert_at_tok,
                    budget_alert_text,
                    pflash_state,
                );
                return;
            }
            // ctx-capacity miss → fall through to default AR body below.
        }
        GenerationRoute::QwenAr
        | GenerationRoute::LlamaAr
        | GenerationRoute::GlimmerAr
        | GenerationRoute::Unknown => {
            // Default AR / unknown body continues below.
            // temp>0 DFlash-disabled visibility (warning text only; route is AR).
            if temp > 1e-6
                && m.speculator.is_some()
                && hipfire_loader::carrier_for(m.arch_id)
                    .map(|c| c.caps().supports_dflash())
                    .unwrap_or(false)
                && !route_inputs.force_ar_chat
            {
                let reason = if route_inputs.temp_spec_env_off {
                    "HIPFIRE_DFLASH_TEMP_SPEC=0"
                } else if min_p.map(|p| p > 0.0).unwrap_or(false) {
                    // Prefer min_p over SWOR-only: selector-chain honors top_p/top_k
                    // but still falls to AR when min_p>0 (DFlash ignores min_p).
                    "request set min_p (sampled DFlash honors top_p/top_k only); AR applies it"
                } else if route_inputs.nonneutral_penalties {
                    "request set a non-neutral repeat/presence/frequency penalty; AR applies it"
                } else if route_inputs.supports_temp_swor
                    && !route_inputs.supports_chain_nucleus_verify
                    && user_explicit_sampling
                {
                    "request set an explicit top_p/top_k/min_p/penalty (ddtree SWOR verify honors temperature only); AR applies them"
                } else if !ngram_can_sample {
                    "loaded drafter is greedy-only (MTP/n-gram); temp>0 runs AR"
                } else if route_inputs.supports_chain_nucleus_verify {
                    "sampled DFlash chain nucleus not engaged (check HIPFIRE_FAST_SAMPLE / temp-spec gates)"
                } else {
                    "ddtree SWOR verify not active (needs ddtree_budget>0)"
                };
                eprintln!(
                    "[hipfire] id={id}: temp>0 DFlash spec disabled -> AR ({reason}). Temperature honored; spec speedup off."
                );
            }
        }
    }

    // Auto-reset on multi-turn rollover. When eviction is active (operator
    // enabled cask_sidecar at load), the physical buffer is bounded by
    // budget+beta+safety regardless of conversation length, so reset never
    // needs to fire — eviction reclaims slots after each token. When eviction
    // is OFF, physical grows unbounded up to max_seq; reset when we'd overrun.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "[qwen-cache GEN-ENTRY] conv_tok={} seq_pos={}",
            m.conversation_tokens.len(),
            m.seq_pos
        );
    }
    if m.eviction.is_none()
        && m.seq_pos
            .saturating_add(prompt_est)
            .saturating_add(max_tokens)
            > m.max_seq
    {
        eprintln!(
            "[daemon] context full ({}/{}) — resetting conversation",
            m.seq_pos, m.max_seq
        );
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        crate::common::free_checkpoints(&mut m.prefill_checkpoints, gpu);
        crate::common::free_checkpoints(&mut m.dflash_checkpoints, gpu);
        // Free the speculator's (relocated) checkpoint ring on reset — this AR
        // path is reachable by a DFlash-capable model (temp>0 / budgeted-think /
        // HIPFIRE_DFLASH_CHAT=0), so its drafter state must not survive here.
        if let Some(s) = m.speculator.as_mut() {
            if let Err(e) = s.reset(gpu) {
                crate::dense::emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("context reset failed: {e}"),
                    "gpu",
                    true,
                    false,
                );
                return;
            }
        }
        // Zero DeltaNet state on reset. qwen35 recurrent state lives in the
        // bundle (ModelState::Qwen35), not the always-None m.dn_state/m.kv_cache.
        // Use the canonical reset so newly added recurrent buffers (notably the
        // Q8 error-feedback residual) cannot leak across rollover boundaries.
        if let Some(b) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) {
            if let Err(e) = b.dn_state.reset(gpu) {
                crate::dense::emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("context reset failed: {e}"),
                    "gpu",
                    true,
                    false,
                );
                return;
            }
            b.kv_cache.compact_offset = 0;
        }
        if let Some(b) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_llama::LlamaBundle>()
        }) {
            b.kv.compact_offset = 0;
        }
        if let Some(ad) = m.kv_adaptive.as_mut() {
            if let Some(b) = m.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
            }) {
                ad.reset_with_cache(gpu, &mut b.kv_cache);
            } else {
                ad.reset();
            }
        }
    }

    // `nl` is needed for the trailer write after natural <|im_end|>
    // termination; `im_end` derives the EOS-check token id. Other
    // ChatML scaffolding tokens are now built inside hipfire_runtime::prompt_frame.
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let raw_q_tokens = tokenizer.encode(prompt);

    // ── PFlash compression (Phase 4.1 #93) ──────────────────────────────
    //
    // Only on first turn (seq_pos == 0). Multi-turn compression of newly-
    // added user content has knock-on effects on prior KV state that we
    // haven't validated yet, so subsequent turns always bypass.
    //
    // Compression operates on the user's actual content tokens
    // (`raw_q_tokens`); chat-template scaffolding (im_start / role / nl /
    // im_end) wraps the result AFTER and is never compressed away.
    // Empty must_keep_spans is correct: there are no chat boundaries
    // INSIDE q_tokens (they live in the scaffolding the daemon adds).
    //
    // Bypass / compressed status is reported as a `pflash_compressed` or
    // `pflash_bypass` event so operators can see what the request actually
    // ran through.
    //
    // Tool-call detection: the prompt may contain a `<tool_call>` token
    // that the parser uses for structure. Compressing those tokens away
    // would corrupt the response shape, so we surface a ToolCall request
    // kind to the gate and let `decide_bypass` reject the request loudly.
    //
    // Two scan locations:
    //   1. raw_q_tokens (the user message itself).
    //   2. system_prompt -- the OpenAI-compatible serve path (the hipfire
    //      CLI's HTTP gateway) puts tool definitions and the `<tool_call>`
    //      format example in the system prompt when `body.tools` is
    //      present. A first-turn user message with tools therefore needs a
    //      system-prompt scan or it would slip through as Text and get its
    //      schema text mangled by compression.
    //
    // Detection is best-effort -- the special-token id is missing on
    // older vocabs, in which case the gate just routes through Text.
    let request_kind = match tokenizer.special_token_id("<tool_call>") {
        Some(tid) => {
            let in_user = raw_q_tokens.iter().any(|&t| t == tid);
            let in_system = system_prompt
                .map(|s| tokenizer.encode(s).iter().any(|&t| t == tid))
                .unwrap_or(false);
            if in_user || in_system {
                hipfire_pflash::pflash::RequestKind::ToolCall
            } else {
                hipfire_pflash::pflash::RequestKind::Text
            }
        }
        None => hipfire_pflash::pflash::RequestKind::Text,
    };

    // Stashed CompressedPrompt summary (when compression actually fired);
    // appended to the `done` event later so a streaming client gets one
    // consolidated line. None means no compression happened on this request.
    let mut pflash_summary: Option<hipfire_pflash::pflash::CompressedPrompt> = None;
    // Bypass reason when compression was attempted but skipped (mode != Off
    // and a drafter was loaded). PRD §3.1 requires "bypass reason if
    // skipped" in the done object.
    let mut pflash_bypass_reason: Option<String> = None;
    // Effective alpha for this request (from cfg if pflash_state is loaded).
    // PRD §3.1 lists alpha as a required done-object field.
    let pflash_alpha: Option<f32> = pflash_cfg.map(|c| c.alpha);
    // Helper: render the JSON field fragment for `done` per PRD §3.1.
    // Three states:
    //   - compressed: full metadata + alpha
    //   - bypass (non-Off, drafter loaded): alpha + bypass_reason
    //   - nothing: empty string so backwards-compatible clients see the
    //     original done shape
    pub fn pflash_done_fragment(
        s: &Option<hipfire_pflash::pflash::CompressedPrompt>,
        bypass_reason: &Option<String>,
        alpha: Option<f32>,
    ) -> String {
        match (s, bypass_reason) {
            (Some(cp), _) => format!(
                r#","pflash":{{"source_tokens":{},"kept_tokens":{},"keep_ratio":{:.6},"alpha":{:.6},"score_ms":{},"total_ms":{},"source_md5":"{}","compressed_md5":"{}"}}"#,
                cp.source_tokens,
                cp.kept_tokens,
                cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32,
                alpha.unwrap_or(0.0),
                cp.timings.score_ms,
                cp.timings.total_ms,
                cp.source_md5,
                cp.compressed_md5,
            ),
            (None, Some(reason)) => format!(
                r#","pflash":{{"bypass_reason":"{}","alpha":{:.6}}}"#,
                reason.replace('"', "'"),
                alpha.unwrap_or(0.0),
            ),
            (None, None) => String::new(),
        }
    }
    if std::env::var("HIPFIRE_PFLASH_DEBUG").is_ok() {
        eprintln!(
            "[pflash] gen: state={} cfg-present seq_pos={} q={} drafter_gpu={}",
            pflash_state.is_some(),
            m.seq_pos,
            raw_q_tokens.len(),
            drafter_gpu.is_some()
        );
    }
    let q_tokens = if let (Some(state), Some(cfg)) = (pflash_state, pflash_cfg) {
        if m.seq_pos == 0 {
            let compress_gpu: &mut rdna_compute::Gpu = drafter_gpu.as_deref_mut().unwrap_or(gpu);
            // Sibling-device drafter: bind its device before compress, then
            // restore the target binding for decode. No-op when shared.
            compress_gpu.bind_thread_or_warn();
            let decision = hipfire_pflash::pflash::maybe_compress_prompt(
                compress_gpu,
                state,
                cfg,
                &raw_q_tokens,
                request_kind,
                &[],
            );
            gpu.bind_thread_or_warn();
            match decision {
                Ok(hipfire_pflash::pflash::PflashDecision::Compressed(cp)) => {
                    eprintln!(
                        "[pflash] COMPRESSED {} -> {} tok dev1 ({}ms)",
                        cp.source_tokens, cp.kept_tokens, cp.timings.total_ms
                    );
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_compressed","id":"{}","source_tokens":{},"kept_tokens":{},"keep_ratio":{:.6},"source_md5":"{}","compressed_md5":"{}","score_ms":{},"select_ms":{},"gather_ms":{},"total_ms":{}}}"#,
                        id,
                        cp.source_tokens,
                        cp.kept_tokens,
                        cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32,
                        cp.source_md5,
                        cp.compressed_md5,
                        cp.timings.score_ms,
                        cp.timings.select_ms,
                        cp.timings.gather_ms,
                        cp.timings.total_ms,
                    );
                    let _ = stdout.flush();
                    let token_ids = cp.token_ids.clone();
                    pflash_summary = Some(cp);
                    token_ids
                }
                Ok(hipfire_pflash::pflash::PflashDecision::Bypass { reason }) => {
                    eprintln!(
                        "[pflash] BYPASS reason={} q={}",
                        reason.as_str(),
                        raw_q_tokens.len()
                    );
                    // Only emit bypass events for non-trivial reasons.
                    // ModeOff is the silent default; nothing to report.
                    if !matches!(reason, hipfire_pflash::pflash::BypassReason::ModeOff) {
                        let r = reason.as_str();
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"pflash_bypass","id":"{}","reason":"{}"}}"#,
                            id,
                            r.replace('"', "'"),
                        );
                        let _ = stdout.flush();
                        // Stash for the `done` object too so a single-line
                        // log scrape sees both the bypass reason and the
                        // request's prefill timings.
                        pflash_bypass_reason = Some(r);
                    }
                    raw_q_tokens
                }
                Err(e) => {
                    eprintln!("[pflash] ERROR compress: {e}");
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_error","id":"{}","reason":"{}"}}"#,
                        id,
                        e.to_string().replace('"', "'"),
                    );
                    let _ = stdout.flush();
                    raw_q_tokens
                }
            }
        } else {
            raw_q_tokens
        }
    } else {
        raw_q_tokens
    };

    // ChatML framing — two paths:
    //
    //   1) `HIPFIRE_JINJA_CHAT=1` AND model carries an embedded chat_template
    //      AND first turn (seq_pos == 0): render through `JinjaChatFrame`
    //      against the upstream HF Jinja template, producing the byte
    //      sequence the model was actually trained on (fixes the "hand-roll
    //      drifted from upstream template" class — XML tool calls on
    //      Qwen3.5/3.6 instead of JSON, `<|im_start|>user` for tool
    //      responses instead of `<|im_start|>tool`, etc.). PFlash
    //      compression is bypassed under Jinja for now (q_tokens not
    //      reusable when the template renders to a String).
    //
    //   2) Default: hand-rolled `prompt_frame::ChatFrame::Plain`
    //      scaffold, byte-identical to today's behavior.
    //
    // Multi-turn (seq_pos > 0) currently always uses path 2 — Jinja
    // single-turn parity is Stage 2; multi-turn message-history state on
    // the daemon side is Stage 2 follow-up.
    //
    // Thinking-off interop with `assistant_prefix`: the CLI sets BOTH
    // `max_think_tokens = 1` AND `assistant_prefix = ClosedThink` when
    // the request asks for non-thinking. The Jinja path keys off
    // `max_think_tokens != 1` for `enable_thinking`; the Plain path
    // honors `assistant_prefix` directly (ClosedThink emits a closed
    // `<think></think>` block after the assistant prefix). Each path
    // picks up the signal it needs.
    // LFM2.5 (arch_id 11) REQUIRES its embedded Jinja chat_template — the
    // hand-rolled Plain ChatML path omits LFM2's `<|startoftext|>` BOS and
    // produces garbage. Force jinja on for arch 11 (falls back to Plain only if
    // the .hfq carries no template, e.g. an older A1B convert).
    // Jinja default-ON (flipped 2026-06-09): render through the model's chat
    // template for ALL arches; opt out with HIPFIRE_JINJA_CHAT=0 (hand-rolled
    // ChatML/Plain). Falls back to Plain automatically when no template resolves.
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
    // Jinja renders the FULL conversation every turn (stateless full-render,
    // like crate::qwen::generate_dflash) — fire on every turn, not just `seq_pos == 0`.
    // `render_messages` below replays `messages_history` (all prior turns) and
    // includes the system prompt, so turn 2+ no longer falls through to the
    // Plain branch (which dropped the system prompt and lost the Jinja
    // template). The cold-reset further down (`jinja_active && seq_pos > 0`)
    // re-prefills this full render from position 0.
    let try_jinja = jinja_enabled && m.chat_template.is_some();
    let mut started_in_think = matches!(
        assistant_prefix,
        hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
    );
    let new_tokens = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort,
        };
        // Phase 1 of Jinja-everywhere migration: when the caller supplies
        // either a `tools` array or a `messages` history (or both), route
        // through `render_messages` so the upstream template's
        // `{% if tools %}` / multi-turn branches fire. With neither
        // supplied, fall through to the single-turn `render()` convenience,
        // which is byte-identical to the synthesized [system?, user]
        // path that shipped under HIPFIRE_JINJA_CHAT=1 before this change.
        let render_result = if tools.is_some() || messages_history.is_some() {
            // Synthesize [system?, user] when no explicit history was
            // provided. Tools-with-legacy-prompt is the natural OpenAI
            // function-calling shape (one turn + tool definitions).
            let synthesized: Vec<hipfire_runtime::prompt_frame::Message>;
            let messages_slice: &[hipfire_runtime::prompt_frame::Message] = match messages_history {
                Some(m) => m,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(hipfire_runtime::prompt_frame::Message {
                            role: hipfire_runtime::prompt_frame::Role::System,
                            content: sys.to_string(),
                            reasoning_content: None,
                            name: None,
                            rendered_name: None,
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                            tool_plan: String::new(),
                        });
                    }
                    v.push(hipfire_runtime::prompt_frame::Message {
                        role: hipfire_runtime::prompt_frame::Role::User,
                        content: prompt.to_string(),
                        reasoning_content: None,
                        name: None,
                        rendered_name: None,
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        tool_plan: String::new(),
                    });
                    synthesized = v;
                    &synthesized
                }
            };
            frame.render_messages(messages_slice, tools, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => {
                // Qwen3's bundled froggeric Jinja owns generation framing.
                // Its rendered tail, not the template-less ChatFrame prefix,
                // determines the initial response channel.
                started_in_think = render_tail_opens_think(&rendered);
                tokenizer.encode(&rendered)
            }
            Err(e) => {
                if reasoning_effort.is_some() {
                    crate::dense::emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &format!("jinja render: {e}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    return;
                }
                eprintln!("[daemon] jinja render failed ({e}) — falling back to Plain");
                hipfire_runtime::prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: "",
                    assistant_prefix,
                    raw: false,
                }
                .build_with_user_tokens(&q_tokens)
            }
        }
    } else {
        hipfire_runtime::prompt_frame::ChatFrame {
            tokenizer,
            system: if m.seq_pos == 0 { system_prompt } else { None },
            user: "", // unused: we pass tokens directly via build_with_user_tokens
            assistant_prefix,
            raw: false,
        }
        .build_with_user_tokens(&q_tokens)
    };

    // ── Prompt cache (LCP-based) — Qwen3.5/3.6 only ──────────────────────
    //
    // Mirrors V4F's prefix-cache (daemon.rs ~5390). Eligible when:
    //   - HIPFIRE_QWEN_PROMPT_CACHE != "0"  (default on)
    //   - messages_history is provided (full-conversation context)
    //   - eviction not active (compact_offset > 0 invalidates the
    //     "conversation_tokens mirrors KV" invariant the cache relies on)
    //   - PFlash compression not enabled this session (compression
    //     changes the KV's token IDs relative to msg.content from history)
    //   - prior conversation_tokens non-empty (first turn = nothing to LCP)
    //
    // On HIT we set `m.seq_pos = LCP` and override `new_tokens` to the
    // suffix slice [LCP..] so the prefill below only writes new tokens.
    // DeltaNet state at position LCP is already correct (cumulative from
    // prior decode). On MISS (divergence in the middle) we full-reset
    // (seq_pos=0, conversation_tokens.clear(), zero DeltaNet, KV
    // compact_offset=0) and prefill the FULL rendered prompt — DeltaNet
    // is not reversible to position M<N so partial rollback is unsafe.
    let cache_kill_switch = std::env::var("HIPFIRE_QWEN_PROMPT_CACHE").ok().as_deref() == Some("0");
    let pflash_active = pflash_cfg
        .map(|c| !matches!(c.mode, hipfire_pflash::pflash::PflashMode::Off))
        .unwrap_or(false);
    // Jinja-on disqualification: when `HIPFIRE_JINJA_CHAT=1` the first
    // turn renders through the upstream HF chat template (which the
    // model was actually trained on — emits default system prompts,
    // Hermes XML tool-call format on Qwen3.5/3.6, etc.). The cache
    // path uses scaffold-style rendering (`ChatScaffold`) which
    // produces a DIFFERENT byte sequence for the same logical content.
    // Mixing the two within a session would degrade output quality
    // (the model sees a different input distribution than it was
    // trained for after turn 1). Skip the cache when Jinja is active
    // so the operator gets consistent rendering across all turns.
    // Cache-with-Jinja is a future project (would require Jinja-side
    // assistant-turn replay).
    let jinja_active = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0")
        && m.chat_template.is_some();
    // Cache-with-Jinja (item #37): `jinja_active` is NO LONGER a disqualifier.
    // When jinja is active the prompt-build below routes through
    // `build_cached_history_jinja` (verbatim assistant-turn splice through the
    // model's trained template) instead of the ChatScaffold `build_cached_history`,
    // so the LCP forward-extension cache now works under HIPFIRE_JINJA_CHAT too.
    let cache_eligible = !cache_kill_switch
        && messages_history.is_some()
        && m.eviction.is_none()
        && !pflash_active
        && !m.conversation_tokens.is_empty();
    if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "[qwen-cache eligible] eligible={} kill={} hist={} evict_none={} !pflash={} jinja={} conv_tok={}",
            cache_eligible, cache_kill_switch, messages_history.is_some(),
            m.eviction.is_none(), !pflash_active, jinja_active, m.conversation_tokens.len(),
        );
    }
    let mut cached_tokens_count: usize = 0;
    let new_tokens: Vec<u32> = if cache_eligible {
        let history = messages_history.unwrap();
        let trace_cache = std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1");
        // Build the canonical full-conversation token stream, replaying
        // any historical assistant turn whose fingerprint matches a
        // cached emission (BPE-bijective replacement).
        let rendered = if jinja_active {
            // Jinja cache (item #37): render the full conversation through the
            // model's trained template, splicing each cached assistant turn's
            // VERBATIM tokens in place of its content (sentinel substitution).
            // The store side (`asst_turn_cache`) holds the GENERATED body only
            // (post-primer); the template renders a history assistant turn as
            // `<|im_start|>assistant\n{content}` with NO generation primer, so
            // we prepend the assistant-opener primer (e.g. `<think>\n`) that
            // THIS turn's cold render emitted — making the spliced stream
            // byte-match `conversation_tokens` for a clean forward extension.
            let primer: Vec<u32> = {
                let im_start = tokenizer.special_token_id("<|im_start|>");
                let opener_len = tokenizer.encode("<|im_start|>assistant\n").len();
                match im_start.and_then(|id| new_tokens.iter().rposition(|&t| t == id)) {
                    Some(q) if q + opener_len <= new_tokens.len() => {
                        new_tokens[q + opener_len..].to_vec()
                    }
                    _ => Vec::new(),
                }
            };
            let template = m.chat_template.as_ref().unwrap();
            let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking,
                bos_token: None,
                reasoning_strength: None,
                reasoning_effort,
            };
            let cache_ref = &mut m.asst_turn_cache;
            let built = hipfire_runtime::prompt_frame::build_cached_history_jinja(
                &frame,
                history,
                tools,
                |msg| {
                    let normalized =
                        crate::common::normalize_asst_turn_for_fingerprint(&msg.content);
                    let fp = crate::common::asst_turn_fingerprint(&normalized, &msg.tool_calls);
                    // Content-only turn: see the dflash sibling above for why `text` is
                    // `msg.content`.
                    let hit = cache_ref.get(&fp).and_then(|turn| {
                        turn.content.as_ref().map(|c| {
                            let mut v = primer.clone();
                            v.extend_from_slice(&c.token_ids);
                            hipfire_runtime::prompt_frame::CachedAssistantTurn {
                                reasoning: None,
                                tools: Vec::new(),
                                content: Some(hipfire_runtime::prompt_frame::CachedAssistantBody {
                                    token_ids: v,
                                    text: msg.content.clone(),
                                }),
                            }
                        })
                    });
                    if trace_cache {
                        eprintln!(
                            "[qwen-cache jinja lookup] fp={:#018x} role={:?} content.len={}/stripped.len={} primer={} hit={}",
                            fp, msg.role, msg.content.len(), normalized.len(), primer.len(), hit.is_some(),
                        );
                    }
                    hit
                },
            );
            match built {
                Ok(t) => t,
                Err(e) => {
                    if reasoning_effort.is_some() {
                        crate::dense::emit_active_attempt_error(
                            stdout,
                            Some(id),
                            &format!("qwen-cache jinja build: {e}"),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        return;
                    }
                    eprintln!("[qwen-cache] jinja cached-history build failed ({e}) — cold render");
                    new_tokens.clone()
                }
            }
        } else {
            let cache_ref = &mut m.asst_turn_cache;
            hipfire_runtime::prompt_frame::build_cached_history(
                tokenizer,
                system_prompt,
                history,
                &q_tokens,
                assistant_prefix,
                crate::qwen::qwen_history_tool_render(&m.model_path),
                |msg| {
                    // Match the store side's stripping. The store applies
                    // `crate::common::strip_think_for_fingerprint` then `maybe_normalize_prompt`
                    // to the model's emitted text before hashing. The CLI
                    // is SUPPOSED to strip `<think>...</think>` from the
                    // visible content before forwarding to clients, but
                    // the inThink state machine only handles paired blocks;
                    // when non-thinking mode prefills `<think>\n\n</think>\n\n`
                    // the model often resumes by emitting another orphan
                    // `</think>\n\n` (training-distribution artifact),
                    // which leaks through to the client's msg.content
                    // verbatim. Apply the same strip here so the lookup
                    // hash matches the store hash regardless of whether
                    // the client preserved the orphan.
                    let stripped = crate::common::strip_think_for_fingerprint(&msg.content);
                    let normalized =
                        hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
                    let fp = crate::common::asst_turn_fingerprint(&normalized, &msg.tool_calls);
                    // `build_cached_history` (the non-Jinja ChatScaffold path) still splices a
                    // flat token vector, so project the content slot out of the per-channel
                    // cache value. Qwen turns are content-only, so nothing is dropped.
                    let hit = cache_ref
                        .get(&fp)
                        .and_then(|turn| turn.content.as_ref().map(|c| c.token_ids.clone()));
                    if trace_cache {
                        eprintln!(
                            "[qwen-cache lookup] fp={:#018x} role={:?} content.len={}/stripped.len={} tool_calls={} hit={}",
                            fp, msg.role, msg.content.len(), normalized.len(),
                            msg.tool_calls.len(), hit.is_some(),
                        );
                    }
                    hit
                },
            )
        };
        // LCP detection vs m.conversation_tokens.
        let prior_len = m.conversation_tokens.len();
        let max_match = prior_len.min(rendered.len());
        let mut lcp = 0usize;
        while lcp < max_match && m.conversation_tokens[lcp] == rendered[lcp] {
            lcp += 1;
        }
        if trace_cache {
            eprintln!(
                "[qwen-cache lcp] prior_len={} rendered_len={} lcp={}",
                prior_len,
                rendered.len(),
                lcp,
            );
            if lcp < prior_len || lcp < rendered.len() {
                // Print full token-ID context on each side past lcp,
                // not just the symmetric overlap window. Lets us see
                // BPE drift cases (same decoded bytes, different ids)
                // and "one side ran out" cases (rendered_len == lcp).
                let pre = lcp.saturating_sub(6);
                let prior_post = (lcp + 16).min(prior_len);
                let rend_post = (lcp + 16).min(rendered.len());
                if lcp > pre {
                    eprintln!(
                        "  common[{}..{}] ids={:?} dec={:?}",
                        pre,
                        lcp,
                        &m.conversation_tokens[pre..lcp],
                        tokenizer.decode(&m.conversation_tokens[pre..lcp]),
                    );
                }
                if prior_post > lcp {
                    eprintln!(
                        "  prior_past[{}..{}] ids={:?} dec={:?}",
                        lcp,
                        prior_post,
                        &m.conversation_tokens[lcp..prior_post],
                        tokenizer.decode(&m.conversation_tokens[lcp..prior_post]),
                    );
                }
                if rend_post > lcp {
                    eprintln!(
                        "  rend_past[{}..{}] ids={:?} dec={:?}",
                        lcp,
                        rend_post,
                        &rendered[lcp..rend_post],
                        tokenizer.decode(&rendered[lcp..rend_post]),
                    );
                }
            }
        } else if lcp < prior_len && prior_len > 50 {
            // Production-visible cache-miss log. Only fires when LCP
            // detected a real divergence (not the first-turn or
            // small-context case). Helps diagnose Pi-style "single-turn
            // cache invalidation" patterns without requiring the
            // operator to reproduce with HIPFIRE_QWEN_CACHE_TRACE=1.
            // Cheap (one eprintln per miss, not per turn).
            //
            // Three windows printed (each clipped to 60 chars):
            //  - common@lcp-4..lcp  — shared tail before divergence
            //  - prior@lcp..lcp+12  — what prior had past lcp (empty if rendered is longer)
            //  - rendered@lcp..lcp+12 — what rendered had past lcp (empty if prior is longer)
            // Plus prior_tail / rendered_tail (last 4 tokens) so we
            // know what each side ends with.
            let pre = lcp.saturating_sub(4);
            let common_dec = if lcp > pre {
                tokenizer.decode(&m.conversation_tokens[pre..lcp])
            } else {
                String::new()
            };
            let prior_post = (lcp + 12).min(prior_len);
            let prior_past_dec = if prior_post > lcp {
                tokenizer.decode(&m.conversation_tokens[lcp..prior_post])
            } else {
                String::new()
            };
            let rend_post = (lcp + 12).min(rendered.len());
            let rend_past_dec = if rend_post > lcp {
                tokenizer.decode(&rendered[lcp..rend_post])
            } else {
                String::new()
            };
            let prior_tail = if prior_len >= 4 {
                tokenizer.decode(&m.conversation_tokens[prior_len - 4..])
            } else {
                tokenizer.decode(&m.conversation_tokens[..])
            };
            let rend_tail = if rendered.len() >= 4 {
                tokenizer.decode(&rendered[rendered.len() - 4..])
            } else {
                tokenizer.decode(&rendered[..])
            };
            eprintln!(
                "[qwen-cache miss] lcp={} prior_len={} rendered_len={}",
                lcp,
                prior_len,
                rendered.len(),
            );
            eprintln!(
                "  common@{}..{}={:?}",
                pre,
                lcp,
                common_dec.chars().take(60).collect::<String>(),
            );
            eprintln!(
                "  prior_past@{}..{}={:?} rendered_past@{}..{}={:?}",
                lcp,
                prior_post,
                prior_past_dec.chars().take(60).collect::<String>(),
                lcp,
                rend_post,
                rend_past_dec.chars().take(60).collect::<String>(),
            );
            eprintln!(
                "  prior_tail={:?} rendered_tail={:?}",
                prior_tail.chars().take(60).collect::<String>(),
                rend_tail.chars().take(60).collect::<String>(),
            );
        }
        if lcp < prior_len || lcp == rendered.len() {
            // Divergence OR exact full-match — NOT a pure forward extension.
            // `lcp == rendered.len()` (⇒ lcp == prior_len) means the request
            // re-renders byte-identically; re-prefilling the final token (the old
            // `lcp-1` over-advance in the else-branch) would re-apply its
            // NON-COMMUTATIVE DeltaNet recurrent update a second time, corrupting
            // S-matrix/conv_state (temp-0 non-determinism + BF16 divergence on
            // re-sent prompts). DeltaNet has no rewindable KV (unlike FullAttention),
            // so the exact-match edge MUST degrade to checkpoint-resume / cold reset —
            // the strict-`<` HIT predicate the sibling DFlash crate::qwen::plan_prompt_cache uses.
            //
            // Divergence: the client sent a non-extension render (it dropped or
            // edited earlier history, so the prior conversation is no longer a
            // prefix of this prompt). Rather than cold-prefill the whole thing,
            // try to RESUME from the latest prefill checkpoint at or before
            // `lcp`: restore the DeltaNet recurrent state captured there, rewind
            // seq_pos + the KV write head, and re-prefill only
            // [resume_pos..rendered.len()). KV for [0..resume_pos] is still
            // resident (positional, never overwritten). Gated to the single-GPU,
            // no-eviction case — eviction remaps physical KV slots, which would
            // invalidate the resident prefix. `seq_pos < rendered.len()` on the
            // chosen checkpoint guarantees ≥1 token is re-prefilled.
            //
            // SAFETY INVARIANT (fix/deltanet-truncation-resume-guard): this
            // restore-checkpoint-at-rpos + replay rendered[rpos..] is exact iff the
            // checkpoint at rpos reflects the committed prefix rendered[..rpos].
            // That holds because (a) rpos <= lcp => rendered[..rpos] ==
            // conversation_tokens[..rpos] (lcp is their longest common prefix), and
            // (b) ALL abort paths now full-reset, so a retained checkpoint can never
            // carry UNCOMMITTED tokens — the poison that used to drift the
            // non-reversible DeltaNet state into garbage. If you ever remove an
            // abort-reset (or let conversation_tokens diverge from the forwarded
            // stream), this resume becomes unsound: re-validate with a per-checkpoint
            // prefix hash (llama.cpp's tokens_hash contract) or cold-recompute.
            // Guarded by scripts/test-qwen35-abort-resume.sh.
            let evict_safe = m.pp <= 1
                && m.eviction.is_none()
                && m.state.as_ref().map_or(true, |s| {
                    let am = s.as_ref();
                    if let Some(b) =
                        (am as &dyn Any).downcast_ref::<hipfire_arch_llama::LlamaBundle>()
                    {
                        b.kv.compact_offset == 0
                    } else if let Some(b) =
                        (am as &dyn Any).downcast_ref::<hipfire_arch_qwen35::Qwen35Bundle>()
                    {
                        // qwen35's KV compact_offset lives in the bundle, not the
                        // always-None m.kv_cache direct field.
                        b.kv_cache.compact_offset == 0
                    } else {
                        true
                    }
                });
            // Resume is only valid for qwen35 (the DeltaNet recurrent state in the
            // bundle). The gate used to read the always-None m.dn_state → resume
            // was silently disabled post-merge; gate on the bundle instead.
            let resume_idx = if ckpt_resume_enabled()
                && evict_safe
                && m.state
                    .as_ref()
                    .map_or(false, |s| s.as_ref().arch_key() == "qwen35")
            {
                m.prefill_checkpoints
                    .iter()
                    .rposition(|(p, _)| *p <= lcp && *p < rendered.len())
            } else {
                None
            };
            let resumed = if let Some(idx) = resume_idx {
                let rpos = m.prefill_checkpoints[idx].0;
                // RESTORE only (do NOT zero): roll the bundle's DeltaNet state
                // back to the checkpoint. Disjoint split: m.state and
                // m.prefill_checkpoints are different fields of `m`.
                let ok = if let (Some(b), Some(ck)) = (
                    m.state.as_mut().and_then(|s| {
                        (s.as_mut() as &mut dyn Any)
                            .downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
                    }),
                    m.prefill_checkpoints.get(idx),
                ) {
                    ck.1.restore_to(&mut b.dn_state, gpu).is_ok()
                } else {
                    false
                };
                if ok {
                    m.seq_pos = rpos;
                    // `evict_safe` guarantees compact_offset == 0, so setting
                    // seq_pos already points the KV write head at rpos — nothing
                    // to restore (checkpoints are only captured with offset 0).
                    m.conversation_tokens.truncate(rpos);
                    truncate_checkpoints(&mut m.prefill_checkpoints, idx + 1, gpu);
                    cached_tokens_count = rpos;
                    eprintln!(
                        "[qwen-cache resume] rewound to checkpoint pos={} (lcp={}, prior_len={}, rendered_len={}) — replaying {} tokens vs cold-prefilling {}",
                        rpos, lcp, prior_len, rendered.len(), rendered.len() - rpos, rendered.len(),
                    );
                    Some(rendered[rpos..].to_vec())
                } else {
                    None
                }
            } else {
                None
            };
            match resumed {
                Some(tail) => tail,
                None => {
                    // No usable checkpoint — full cold reset. DeltaNet recurrent
                    // state is non-reversible; treat as a miss. Inlined (not
                    // `full_reset_cold`) because a `&tokenizer` borrow of `m` is
                    // live here; these are disjoint field accesses. qwen35 state
                    // lives in the bundle (ModelState::Qwen35), not the always-None
                    // m.dn_state/m.kv_cache.
                    m.seq_pos = 0;
                    m.conversation_tokens.clear();
                    crate::common::free_checkpoints(&mut m.prefill_checkpoints, gpu);
                    if let Some(b) = m.state.as_mut().and_then(|s| {
                        (s.as_mut() as &mut dyn Any)
                            .downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
                    }) {
                        let dn = &b.dn_state;
                        for s in &dn.s_matrices {
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                        }
                        for s in &dn.s_scales {
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                        }
                        for s in &dn.conv_states {
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                        }
                        for s in &dn.s_ef_residual {
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                        }
                    }
                    if let Some(b) = m.state.as_mut().and_then(|s| {
                        (s.as_mut() as &mut dyn Any)
                            .downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
                    }) {
                        b.kv_cache.compact_offset = 0;
                    }
                    if let Some(b) = m.state.as_mut().and_then(|s| {
                        (s.as_mut() as &mut dyn Any)
                            .downcast_mut::<hipfire_arch_llama::LlamaBundle>()
                    }) {
                        b.kv.compact_offset = 0;
                    }
                    rendered
                }
            }
        } else {
            // Pure forward extension: `lcp == prior_len && lcp < rendered.len()`.
            // The prior turn left the recurrent DeltaNet state at exactly
            // `prior_len`, so reusing KV/DeltaNet[0..lcp] and prefilling the new
            // suffix `rendered[lcp..]` (≥1 token, since lcp < rendered.len())
            // advances the state correctly with no rewind and no over-advance.
            // The exact-match edge (lcp == rendered.len()) no longer reaches here —
            // it degrades to checkpoint-resume / cold reset above.
            m.seq_pos = lcp;
            cached_tokens_count = lcp;
            rendered[lcp..].to_vec()
        }
    } else {
        new_tokens
    };

    // Jinja path renders the full conversation each turn. When the LCP cache
    // ran this turn (`cache_eligible`), it already managed seq_pos — set it to
    // the LCP on a forward-extension HIT, or full-reset on a MISS — so we must
    // NOT blanket-reset here (that would discard a valid cache hit and force a
    // cold re-prefill every turn). Only cold-reset when the cache did NOT run
    // (item #37): first turn (empty conversation), kill switch
    // (HIPFIRE_QWEN_PROMPT_CACHE=0), eviction/PFlash active. On turn 2+ in those
    // cases, reset BEFORE the budget guard + prefill so the full render writes
    // from position 0 rather than appending to the prior turn's dirty
    // DeltaNet/KV/checkpoint state. Uses `crate::common::free_checkpoints` (NOT a bare
    // `.clear()`) so the checkpoint GPU buffers are freed rather than leaked.
    if jinja_active && !cache_eligible && m.seq_pos > 0 {
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        crate::common::free_checkpoints(&mut m.prefill_checkpoints, gpu);
        crate::common::free_checkpoints(&mut m.dflash_checkpoints, gpu);
        // Free the speculator's (relocated) checkpoint ring on reset — this AR
        // path is reachable by a DFlash-capable model.
        if let Some(s) = m.speculator.as_mut() {
            if let Err(e) = s.reset(gpu) {
                crate::dense::emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("prompt-cache reset failed: {e}"),
                    "gpu",
                    true,
                    false,
                );
                return;
            }
        }
        // qwen35 recurrent state lives in the bundle (ModelState::Qwen35), not
        // the always-None m.dn_state/m.kv_cache. Inlined (disjoint field access)
        // because a `&tokenizer` borrow of `m` is live here.
        if let Some(b) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) {
            let dn = &b.dn_state;
            for s in &dn.s_matrices {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &dn.s_scales {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &dn.conv_states {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &dn.s_ef_residual {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
        }
        if let Some(b) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) {
            b.kv_cache.compact_offset = 0;
        }
        if let Some(b) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_llama::LlamaBundle>()
        }) {
            b.kv.compact_offset = 0;
        }
    }

    // KV-budget guard. Without eviction the physical buffer is the hard cap;
    // we must fit prefill + generation + trailer in one allocation. With
    // eviction, physical is bounded by physical_cap regardless of total tokens
    // — the chunked prefill below calls maybe_evict between chunks, and the
    // decode loop evicts after every token. The only ceiling under eviction is
    // the advertised context window (max_seq) — refuse requests that would
    // overflow it in absolute position terms (current absolute + new).
    let trailer = nl.len();
    let absolute_pos = m.seq_pos.saturating_add(
        m.state
            .as_ref()
            .and_then(|s| {
                let am = s.as_ref();
                if let Some(b) = (am as &dyn Any).downcast_ref::<hipfire_arch_llama::LlamaBundle>()
                {
                    Some(b.kv.compact_offset)
                } else if let Some(b) =
                    (am as &dyn Any).downcast_ref::<hipfire_arch_qwen35::Qwen35Bundle>()
                {
                    // qwen35 KV compact_offset lives in the bundle, not the
                    // always-None m.kv_cache direct field.
                    Some(b.kv_cache.compact_offset)
                } else {
                    None
                }
            })
            .unwrap_or(0),
    );
    if m.eviction.is_none() {
        if m.seq_pos
            .saturating_add(new_tokens.len())
            .saturating_add(max_tokens)
            .saturating_add(trailer)
            > m.physical_cap
        {
            crate::dense::emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > physical_cap={} — reload model with a larger max_seq", m.seq_pos, new_tokens.len(), max_tokens, trailer, m.physical_cap),
                "context_length",
                false,
                false
            );
            let _ = stdout.flush();
            return;
        }
    } else if absolute_pos
        .saturating_add(new_tokens.len())
        .saturating_add(max_tokens)
        .saturating_add(trailer)
        > m.max_seq
    {
        crate::dense::emit_active_attempt_error(
            stdout,
            Some(id),
            &format!("request exceeds advertised context window: absolute={} + prefill={} + max_tokens={} + trailer={} > max_seq={}", absolute_pos, new_tokens.len(), max_tokens, trailer, m.max_seq),
            "context_length",
            false,
            false
        );
        let _ = stdout.flush();
        return;
    }

    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };
    // Special-token attractor blocking (#111). Resolve the token IDs once;
    // each pair is `Some` only when the tokenizer registers both opener
    // and closer as single special tokens (Qwen3+ vocabs). Older vocabs
    // return `None` and the block is silently skipped — no behavior
    // change.
    let tool_call_pair = match (
        tokenizer.special_token_id("<tool_call>"),
        tokenizer.special_token_id("</tool_call>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };
    let think_pair = match (
        tokenizer.special_token_id("<think>"),
        tokenizer.special_token_id("</think>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };
    let prefill_tokens = new_tokens.len();
    // Pure arch→contract selection (same function tests exercise).
    // Qwen AR (5/6) advertises v2; DS4 and others stay unset.
    let gen_contract = crate::common::gen_start_contract_version_for_arch(m.arch_id);
    emit_gen_start(stdout, id, started_in_think, gen_contract);
    let t0 = Instant::now();

    if hipfire_loader::carrier_for(m.arch_id)
        .map(|c| c.caps().has_deltanet)
        .unwrap_or(false)
    {
        // Qwen3.5 / Qwen3.5-MoE — multi-turn: prefill only the NEW turn tokens,
        // continuing from m.seq_pos (KV cache + DeltaNet state are cumulative)
        let b = m
            .state
            .as_mut()
            .and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
            })
            .unwrap();
        let config = &b.config;
        let weights = &b.weights;
        let scratch = &b.scratch;
        let kv = &mut b.kv_cache;
        let dn = &mut b.dn_state;
        // Cold-reset after uncommitted AR prefill/decode failure. Mirrors
        // multi-GPU `reset_pp_uncommitted_state!` and the prefill-abort path:
        // DN is non-reversible, so partial forward must not poison the next
        // turn. Adaptive poison stays sticky (`clear_adaptive_poison: false`).
        macro_rules! reset_ar_uncommitted_state {
            () => {{
                debug_assert!(qwen_ar_forward_fail_action().reset_uncommitted_state);
                debug_assert!(!qwen_ar_forward_fail_action().clear_adaptive_poison);
                for s in &dn.s_matrices {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
                for s in &dn.s_scales {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
                for s in &dn.conv_states {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
                for s in &dn.s_ef_residual {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
                kv.compact_offset = 0;
                if let Some(ad) = m.kv_adaptive.as_mut() {
                    if !ad.is_poisoned() {
                        ad.reset_with_cache(gpu, kv);
                    }
                }
                m.seq_pos = 0;
                m.conversation_tokens.clear();
                crate::common::free_checkpoints(&mut m.prefill_checkpoints, gpu);
            }};
        }

        // Prefill this turn's tokens via the batched prefill entry point.
        // On gfx11+ for MQ4/HFQ4/MQ6/HFQ6 weights this hits the WMMA GEMM
        // fast path; other archs fall back to dp2 / FP16-packed / scalar
        // variants. The one sequential hotspot inside is the gated_delta_net
        // Q8 state update (N sequential per-token calls per LA layer, byte-
        // exact with decode to keep the quality gate green).
        //
        // Note: forward_prefill_batch launches HIP kernels asynchronously.
        // The t_prefill mark below lives AFTER the first sample_top_p, whose
        // D2H readback of tok0 forces a device sync — that's the point at
        // which the first token is actually ready to stream. Placing the
        // mark earlier captures CPU-dispatch time, which under-reports
        // prefill by a large factor (prefill_tok_s ~5–10× too optimistic).
        //
        // Under eviction: chunk prefill to the (budget+beta) eviction window
        // and call `maybe_evict` between chunks so physical never exceeds
        // physical_cap. Chunk size caps out at physical capacity available —
        // when physical is at post-evict `budget`, a full `beta`-sized chunk
        // can run before the next eviction fires.
        // Prefill loop with abort support. The CLI sends
        // `{type:"abort","id":"..."}` when the HTTP client closes the
        // connection (curl `-m` timeout, Pi/opencode response timer
        // fired, etc.); the stdin reader thread sets the abort flag
        // and the chunk loop below picks it up. The no-eviction path
        // is manually chunked at the ordinary prefill outer bound so abort
        // latency is bounded to one chunk (~5 s on gfx1151 at 50 tps).
        //
        // On abort, DeltaNet's non-reversible state means we can't
        // rewind to the pre-prefill position — full reset (seq_pos=0,
        // conversation_tokens cleared, DN s/conv buffers zeroed,
        // KV compact_offset=0). Next request hits cache miss and
        // does a full re-prefill from scratch, which is the same cost
        // as letting the abandoned prefill drain — but the client
        // gets control back immediately instead of waiting.
        let mut prefill_aborted = false;
        if let Some(ref ev) = m.eviction {
            let window = ev.budget() + ev.beta();
            let mut remaining: &[u32] = &new_tokens;
            while !remaining.is_empty() {
                if check_abort(id) {
                    prefill_aborted = true;
                    break;
                }
                let adaptive_staging = m
                    .kv_adaptive
                    .as_ref()
                    .is_some_and(|ad| !ad.handoff_complete());
                let chunk_limit =
                    qwen_ar_eviction_prefill_chunk_limit(m.seq_pos, window, adaptive_staging);
                let chunk_len = remaining.len().min(chunk_limit);
                let (chunk, rest) = remaining.split_at(chunk_len);
                // Outer eviction-window / maybe_evict cadence is unchanged;
                // only internal temporary PBS/chunks stay at the historical
                // memory-safe ceiling (PREFILL_MAX_BATCH=256).
                if let Err(e) = qwen35::forward_prefill_batch_capped(
                    gpu,
                    weights,
                    config,
                    chunk,
                    m.seq_pos,
                    kv,
                    dn,
                    scratch,
                    None,
                    None,
                    None,
                    None,
                    qwen35::PREFILL_MAX_BATCH,
                ) {
                    let action = qwen_ar_forward_fail_action();
                    if action.reset_uncommitted_state {
                        reset_ar_uncommitted_state!();
                    }
                    if action.emit_request_error {
                        write_error(
                            stdout,
                            id,
                            &qwen_ar_forward_fail_message("forward_prefill_batch", e),
                        );
                    }
                    return;
                }
                m.seq_pos += chunk_len;
                // The eviction gate remains closed until adaptive KV reaches
                // its explicit handoff point and finishes every transcode.
                // Drive that transition at the same bounded committed-prefix
                // boundaries as the ordinary adaptive prefill path, before
                // asking eviction to observe the gate.
                if let Some(ad) = m.kv_adaptive.as_mut() {
                    match ad.maybe_downshift(gpu, kv, m.seq_pos) {
                        Ok(steps) => {
                            for step in steps {
                                eprintln!(
                                    "[adaptive-kv] downshift @ pos {} (eviction prefill): {:?} (K={:?} V={:?})",
                                    m.seq_pos, step, ad.cur_k, ad.cur_v
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "[adaptive-kv] maybe_downshift error @ pos {} (eviction prefill): {:?} — poisoning model",
                                m.seq_pos, e
                            );
                            reset_ar_uncommitted_state!();
                            crate::dense::emit_active_attempt_error(
                                stdout,
                                Some(id),
                                &format!(
                                    "adaptive KV transition failed during eviction prefill: {e}"
                                ),
                                "transient",
                                true,
                                false,
                            );
                            let _ = stdout.flush();
                            return;
                        }
                    }
                }
                if let Some(hipfire_runtime::triattn::EvictionResult {
                    new_physical: new_phys,
                    ..
                }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap()
                {
                    m.seq_pos = new_phys;
                }
                remaining = rest;
            }
        } else {
            // Manually chunk the no-eviction prefill so the abort check fires
            // between batches. Outer chunks must agree with internal chunk /
            // PBS capacity via `prefill_max_batch` (gfx1201 defaults 384;
            // gfx11/CDNA stay 256; HIPFIRE_PREFILL_MAX_BATCH>=2 overrides).
            // Adaptive-KV keeps the hard `PREFILL_MAX_BATCH` (256) cap so the
            // controller's margin and maybe_downshift boundaries stay exact.
            let chunk_max = if m.kv_adaptive.is_some() {
                qwen35::PREFILL_MAX_BATCH
            } else {
                qwen35::prefill_max_batch(gpu)
            };
            let mut start = 0usize;
            while start < new_tokens.len() {
                if check_abort(id) {
                    prefill_aborted = true;
                    break;
                }
                let end = (start + chunk_max).min(new_tokens.len());
                let chunk = &new_tokens[start..end];
                if let Err(e) = qwen35::forward_prefill_batch(
                    gpu, weights, config, chunk, m.seq_pos, kv, dn, scratch, None, None, None, None,
                ) {
                    let action = qwen_ar_forward_fail_action();
                    if action.reset_uncommitted_state {
                        reset_ar_uncommitted_state!();
                    }
                    if action.emit_request_error {
                        write_error(
                            stdout,
                            id,
                            &qwen_ar_forward_fail_message("forward_prefill_batch", e),
                        );
                    }
                    return;
                }
                m.seq_pos += chunk.len();
                // Adaptive KV: downshift BETWEEN prefill chunks the moment the
                // start-tier (q8/fwht4) buffer fills, so a long prompt can't
                // overflow the floor-sized buffer before decode begins. The
                // controller's margin (>= PREFILL_MAX_BATCH) guarantees the chunk
                // that trips a threshold still wrote in-bounds; this call then
                // re-quantizes [0, seq_pos) down a tier, freeing room for the next
                // chunk. `m.kv_adaptive` is disjoint from the live kv/dn borrows.
                if let Some(ad) = m.kv_adaptive.as_mut() {
                    match ad.maybe_downshift(gpu, kv, m.seq_pos) {
                        Ok(steps) => {
                            for step in steps {
                                eprintln!(
                                    "[adaptive-kv] downshift @ pos {} (prefill): {:?} (K={:?} V={:?})",
                                    m.seq_pos, step, ad.cur_k, ad.cur_v
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "[adaptive-kv] maybe_downshift error @ pos {} (prefill): {:?} — poisoning model",
                                m.seq_pos, e
                            );
                            // maybe_downshift already poisons on partial failure; surface hard.
                            crate::dense::emit_active_attempt_error(
                                stdout,
                                Some(id),
                                &format!("adaptive KV transition failed during prefill: {e}"),
                                "transient",
                                true,
                                false,
                            );
                            let _ = stdout.flush();
                            return;
                        }
                    }
                }
                // Snapshot the recurrent state every ckpt_interval() tokens so a
                // later divergent render can resume here instead of cold. `dn`
                // (&mut m.dn_state) and &mut m.prefill_checkpoints are disjoint
                // fields, so this composes with the live kv/dn borrows.
                if ckpt_resume_enabled() {
                    speculative::take_dn_checkpoint(
                        &mut m.prefill_checkpoints,
                        dn,
                        gpu,
                        m.seq_pos,
                        ckpt_interval(),
                        ckpt_max(),
                    );
                }
                start = end;
            }
        }
        if prefill_aborted {
            // Drop live bundle borrows (kv/dn) before production rollback, which
            // reborrows m.state for the authoritative fail-closed reset+sync.
            let _ = (kv, dn, weights, config, scratch);
            let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
            crate::common::emit_spec_cancel_after_rollback(stdout, id, 0, &ep);
            return;
        }
        // Adaptive KV: after prefill, downshift any tiers whose threshold the
        // prefill already crossed (so the q8/start buffer never overflows before
        // decode starts). `kv` (=m.kv_cache) and m.kv_adaptive are distinct
        // fields → NLL splits the borrow.
        if let Some(ad) = m.kv_adaptive.as_mut() {
            match ad.maybe_downshift(gpu, kv, m.seq_pos) {
                Ok(applied) => {
                    for step in &applied {
                        eprintln!(
                            "[adaptive-kv] downshift @ pos {}: {:?} (K={:?} V={:?})",
                            m.seq_pos, step, ad.cur_k, ad.cur_v
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[adaptive-kv] maybe_downshift error @ pos {} (post-prefill): {:?} — poisoning model",
                        m.seq_pos, e
                    );
                    crate::dense::emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &format!("adaptive KV transition failed after prefill: {e}"),
                        "transient",
                        true,
                        false,
                    );
                    let _ = stdout.flush();
                    return;
                }
            }
        }
        m.conversation_tokens.extend_from_slice(&new_tokens);

        // serve-fault-inject: one-shot after GPU/KV/recurrent mutation, before
        // any token/reasoning/tool_calls/commit_ready visibility. Only drop
        // live bundle borrows when the fault actually fires (rollback needs
        // &mut m.state).
        #[cfg(feature = "serve-fault-inject")]
        if FAULT_AFTER_PREFILL_ARMED.with(|c| c.get()) && matches!(m.arch_id, 5 | 6) {
            let _ = take_fault_after_prefill();
            let _ = (kv, dn, weights, config, scratch);
            let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
            crate::common::emit_fail_closed_error(
                stdout,
                Some(id),
                "injected fault after prefill",
                "gpu",
                true,
                &ep,
            );
            return;
        }

        // ngram scope for the repeat penalty: ONLY generated tokens (never the
        // prompt). Prior design included the user's prompt as an anti-loop
        // anchor, but that penalizes the very tokens we're asked to recall
        // (names, numbers, facts) under MQ4/MQ6 quantizations that are more
        // RP-sensitive than llama.cpp's Q4_K. First sample: empty scope (no
        // generated tokens yet); subsequent samples: generated-so-far only.
        let ngram_scope_start = m.conversation_tokens.len();
        // Boundary marker for the prompt-cache: the model's verbatim
        // emitted tokens start here. Used after the decode loop to
        // slice out cached_seq for `asst_turn_cache`. Equal to
        // ngram_scope_start by construction; aliased for readability.
        let decode_start_tokens_idx = ngram_scope_start;

        // Generate. GPU-side sampling eliminates per-token logits download +
        // CPU softmax + CPU repeat penalty. Closes the 2× gap between raw
        // bench throughput and daemon throughput.
        //
        // Kernel signature reads `repeat_tokens[0..repeat_window]`, so we
        // only need to upload the tokens that will actually be read — no
        // need to clear the buffer between calls. The upload is on the same
        // stream as the sample kernel launch, so the copy and compute pipeline
        // naturally.
        let vocab_size = config.vocab_size;
        let mut rng_state: u32 = 0x13579BDFu32;
        // Effective penalty window = request `repeat_window` (default 128),
        // bounded by the GPU repeat_buf capacity (2048). The buffer is sized
        // large so presence/frequency penalties CAN use a wider window when a
        // request asks for it, but the default stays at the historical 128 —
        // we do NOT widen the repeat-penalty window for all traffic.
        let repeat_buf_cap = (scratch.repeat_buf.buf.size() / 4).min(repeat_window.max(1));

        // Build the list of paired (open, close) attractor pairs once;
        // sampler::collect_unclosed_attractor_blocks decides per-call
        // which openers (if any) trip the depth threshold.
        let attractor_pairs: Vec<(u32, u32)> = tool_call_pair
            .into_iter()
            .chain(think_pair.into_iter())
            .collect();

        // ── Grammar-guided decoding setup ───────────────────────────
        //
        // When the request carries tools, build a qwen35 grammar matcher
        // and pin a vocab-sized decoded-text vector for mask construction.
        // The matcher constrains sample-time logits the moment the model
        // commits to `<tool_call>` — preventing the qwen3.6:27b "ChatML
        // noise as tool_call body" attractor observed in Pi turn 12 (the
        // model emitted `<|im_start|>assistant "..."}}` between the open
        // and close tags, breaking JSON parse → daemon emitted
        // `finish_reason: "stop"` with garbage content → Pi agent loop
        // terminated). See `crates/hipfire-arch-qwen35/src/grammar.rs`
        // for the state machine and the V4F path
        // (`crates/hipfire-arch-deepseek4/src/grammar.rs`) for the
        // structurally-similar DSML grammar.
        //
        // Disable with `HIPFIRE_QWEN35_GRAMMAR=0` for A/B comparison.
        let grammar_enabled = hipfire_runtime::prompt_frame::qwen35_grammar_on(
            std::env::var("HIPFIRE_QWEN35_GRAMMAR").ok().as_deref(),
            &m.model_path,
        );
        let tool_schemas_qwen: Vec<saddle_core::grammar::json::ToolSchema> = if grammar_enabled {
            tools
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| {
                            let func = t.get("function").unwrap_or(t);
                            let name = func
                                .get("name")
                                .and_then(|v| v.as_str())
                                .filter(|s| !s.is_empty())?
                                .to_string();
                            let required: Vec<String> = func
                                .get("parameters")
                                .and_then(|p| p.get("required"))
                                .and_then(|r| r.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|v| v.as_str().map(String::from))
                                        .collect()
                                })
                                .unwrap_or_default();
                            Some(saddle_core::grammar::json::ToolSchema { name, required })
                        })
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let grammar_active = !tool_schemas_qwen.is_empty();
        let mut grammar_matcher = saddle_core::grammar::json::Matcher::with_config(
            tool_schemas_qwen,
            hipfire_loader::carrier_for(m.arch_id)
                .map(|c| c.grammar_config())
                .unwrap_or_default(),
        );
        // One-time vocab decode for token mask construction. Reuses the
        // model-level cache so subsequent requests on the same model skip
        // the ~150k-entry decode.
        let qwen_grammar_vocab: Option<std::sync::Arc<Vec<String>>> = if grammar_active {
            if m.decoded_vocab.is_none() {
                let n = tokenizer.vocab_size();
                let v: Vec<String> = (0..n).map(|id| tokenizer.decode(&[id as u32])).collect();
                m.decoded_vocab = Some(std::sync::Arc::new(v));
            }
            m.decoded_vocab.clone()
        } else {
            None
        };
        let empty_vocab: Vec<String> = Vec::new();
        let grammar_vocab: &[String] = qwen_grammar_vocab
            .as_deref()
            .map(|v| v.as_slice())
            .unwrap_or(&empty_vocab);
        let mut grammar_mask: Vec<bool> = vec![true; grammar_vocab.len()];

        // First sample: use conversation so far as scope.
        let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
        // #111 attractor block: empty `ngram_scope` on first sample (no
        // generated tokens yet), so the unclosed-depth is always 0 and
        // `blocked` is empty. Still call collect_* for symmetry with
        // the loop body, in case a future change moves this block into
        // a multi-step warmup.
        let mut blocked0: Vec<u32> = Vec::new();
        sampler::collect_unclosed_attractor_blocks(
            ngram_scope,
            &attractor_pairs,
            20,
            2,
            &mut blocked0,
        );
        let cfg0 = SamplerConfig {
            temperature: temp,
            top_p,
            repeat_penalty,
            // Window is bounded by the GPU repeat_buf capacity (sized
            // at 64 in ForwardScratch::new). Pre-PR3 code did this
            // bound by setting `scope_start = len - repeat_buf_cap`
            // and passing `scope.len()` to the kernel; we let
            // sampler::sample do the same `min(window, buf_cap)`
            // internally.
            repeat_window: repeat_buf_cap,
            presence_penalty,
            frequency_penalty,
            blocked_tokens: blocked0,
            top_k,
            min_p,
        };
        // Grammar-gated sample: GPU fast path when the matcher is free
        // (the common case — no tool_call mid-flight); CPU slow path when
        // the matcher is constraining, so we can apply the token mask to
        // the logits before sampling. See setup block above for rationale.
        let tok0 = if grammar_active && !grammar_matcher.is_free() {
            let mut logits = gpu
                .download_f32(&scratch.logits)
                .unwrap_or_else(|_| vec![0.0f32; vocab_size]);
            grammar_matcher.token_mask(grammar_vocab, &mut grammar_mask);
            saddle_core::grammar::json::Matcher::apply_mask_to_logits(&grammar_mask, &mut logits);
            sampler::sample_cpu(&mut logits, ngram_scope, &cfg0)
        } else {
            sampler::sample(
                gpu,
                &scratch.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                vocab_size,
                ngram_scope,
                &cfg0,
                &mut rng_state,
            )
        };
        if grammar_active {
            let text = tokenizer.decode(&[tok0]);
            grammar_matcher.advance(&text);
        }
        // First token is ready (sample_top_p's D2H forces GPU sync). This is
        // the user-observable "time to first token" boundary — prefill above,
        // decode loop below.
        let t_prefill = Instant::now();
        let mut next_token = tok0;

        let mut generated = 0;
        let mut streamed_tokens: Vec<u32> = Vec::new();
        let mut bytes_fed_to_filter = 0usize;
        // Increment-A semantic producer: sole authority for client-visible text
        // and structured tool_calls on this AR path. Raw token commit stays
        // upstream via `commit_and_observe` (conversation_tokens / streamed /
        // seq_pos advance before classify).
        let mut semantic =
            QwenArSemanticProducer::new_with_tool_protocol(id, started_in_think, tools_nonempty);
        let mut alert_fired = false;
        // max_think_tokens enforcement state. think_count increments only
        // while we observe ourselves to be inside a `<think>...</think>`
        // block via the same decoded-text scan budget_alert uses. When the
        // cap is hit we splice "</think>\n" into the stream (KV write +
        // stdout emit + advance generated) so the model finishes thinking
        // and commits to an answer with the remaining max_tokens budget.
        // Re-armable: if the model later opens another <think> in the same
        // turn (rare) the counter resets and the cap re-fires.
        let mut think_count: usize = 0;
        let mut prev_in_think: bool = false;
        // Force-answer is a ONE-SHOT signal (check_force_answer clears on read),
        // but 35b-a3b re-opens <think> after a forced close and then thinks
        // unbounded until the client times out. Latch it for the rest of the
        // turn: a re-opened <think> is re-closed, and (for single-token
        // think-open vocabs) the open token is blocked outright so the model
        // commits to its answer instead of looping back into thinking.
        let mut force_answer_latched = false;
        let think_open_tok = tokenizer.special_token_id("<think>");
        // Hard bound on TOTAL thinking across the turn (re-arm-proof, unlike the
        // per-block max_think_tokens which resets on each re-opened <think>).
        // 0 = off. At the cap we force-close + block <think> (best effort to make
        // the model answer); if it's STILL thinking a margin past the cap, we
        // force EOS so the turn can't run unbounded — 35b-a3b re-opens <think>
        // after the one-shot force-answer and out-thinks client timeouts.
        let max_total_think = hipfire_runtime::config::get().max_total_think_tokens;
        let mut total_think_tokens: usize = 0;
        // Post-latch answer bound (see the _multi decode path for rationale): the
        // +256 EOS below only counts in-think tokens, so a non-think ramble or a
        // re-open loop after the cap latches would run to max_tokens. Hard-EOS
        // once generation runs this many tokens past the latch.
        let post_latch_answer_budget: usize = std::env::var("HIPFIRE_POST_LATCH_ANSWER_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(768);
        let mut latch_gen_mark: Option<usize> = None;

        // N-gram loop detector: track 4-gram token sequences. When any
        // 4-gram repeats more than `ngram_loop_threshold` times in the
        // last `ngram_window` tokens, force EOS. This catches answer-phase
        // repetition loops that the think cap and repeat penalty miss.
        // Operates on token IDs (no decode overhead).
        // Implementation lives in `hipfire_runtime::loop_guard`; defaults read from
        // HIPFIRE_NGRAM_LOOP_THRESHOLD (default 8, 0 = disabled) and
        // HIPFIRE_NGRAM_WINDOW (default 256). See loop_guard.rs.
        let loop_guard =
            hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get());

        // `while` instead of `for 0..max_tokens` so budget-alert injection
        // (which increments `generated` beyond the iteration count) can't
        // push generated past max_tokens: each loop start rechecks the cap.
        while generated < max_tokens {
            // Decode-side abort check. Client cancel (Pi 4-min idle
            // timeout firing while the CLI buffers tokens for tool-call
            // detection — wire shows zero output until `done`) sends
            // `{type:"abort","id":"..."}` over stdin; the reader thread
            // latches abort on the active terminal-control transaction and we bail at the next iteration.
            // Emit aborted+done so the CLI's drain loop terminates
            // cleanly without an extra max_tokens worth of wasted decode.
            if check_abort(id) {
                // Drop live bundle borrows before production fail-closed rollback.
                // Tokens so far are uncommitted; DN is non-reversible — full
                // cold reset via the common attestation path (not a partial
                // manual clear). Terminal is aborted+done only when attested.
                let _ = (kv, dn, weights, config, scratch);
                let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
                crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
                return;
            }
            // Write this token's K/V to the cache BEFORE any client-visible
            // emit so a VMM map/growth failure cannot stream a token whose
            // trunk write never committed. Successful path still advances
            // conversation/stream state and emits with the same semantics as
            // before (generated++, push, committed, token text).
            //
            // Under eviction, m.seq_pos is the *physical* write slot; we
            // advance and call maybe_evict immediately so the next write
            // never overruns physical_cap. compact_offset bookkeeping on
            // the cache itself keeps RoPE phase correct across evictions.
            if let Err(e) = qwen35::forward_scratch(
                gpu, weights, config, next_token, m.seq_pos, kv, dn, scratch,
            ) {
                let action = qwen_ar_forward_fail_action();
                debug_assert!(!action.emit_failed_token);
                if action.reset_uncommitted_state {
                    reset_ar_uncommitted_state!();
                }
                if action.emit_request_error {
                    write_error(
                        stdout,
                        id,
                        &qwen_ar_forward_fail_message("forward_scratch decode", e),
                    );
                }
                return;
            }
            generated += 1;
            // Incremental UTF-8 + filter routing via producer-owned
            // commit-then-classify. Raw commit (conversation/stream/seq_pos)
            // runs inside the closure before fallible classify; decode delta
            // is computed after the real push.
            let prev_fed = bytes_fed_to_filter;
            let elapsed_ms = t0.elapsed().as_millis() as u64;
            match semantic.commit_and_classify(
                stdout,
                next_token,
                || {
                    let pos = qwen_ar_raw_commit_token(
                        &mut m.conversation_tokens,
                        &mut streamed_tokens,
                        &mut m.seq_pos,
                        next_token,
                        QwenArRawCommitDisposition::ClassifiedVisible,
                    );
                    let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
                    let new_bytes = all_bytes[prev_fed..].to_vec();
                    bytes_fed_to_filter = all_bytes.len();
                    (pos, new_bytes)
                },
                |pos, out| {
                    crate::common::emit_committed_event(out, id, next_token, pos, elapsed_ms);
                },
            ) {
                Ok(true) => break, // decoded EOT suppressed; stop generation
                Ok(false) => {}
                Err(err) => {
                    emit_error_with_id(stdout, id, &err.to_string());
                    return;
                }
            }
            // Checkpoint during decode too, so a long generated turn (e.g. a
            // big code emission) can be resumed mid-region if the NEXT turn's
            // render diverges within it — without replaying the whole
            // generation. No-op under eviction (compact_offset != 0).
            if ckpt_resume_enabled() {
                speculative::take_dn_checkpoint(
                    &mut m.prefill_checkpoints,
                    dn,
                    gpu,
                    m.seq_pos,
                    ckpt_interval(),
                    ckpt_max(),
                );
            }
            if let Some(ref ev) = m.eviction {
                if let Some(hipfire_runtime::triattn::EvictionResult {
                    new_physical: new_phys,
                    ..
                }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap()
                {
                    m.seq_pos = new_phys;
                }
            }
            // Adaptive KV: downshift K/V precision as seq_pos crosses capacity
            // thresholds. `kv` (=m.kv_cache) and m.kv_adaptive are distinct
            // fields → NLL splits the borrow.
            if let Some(ad) = m.kv_adaptive.as_mut() {
                match ad.maybe_downshift(gpu, kv, m.seq_pos) {
                    Ok(applied) => {
                        for step in &applied {
                            eprintln!(
                                "[adaptive-kv] downshift @ pos {}: {:?} (K={:?} V={:?})",
                                m.seq_pos, step, ad.cur_k, ad.cur_v
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "[adaptive-kv] maybe_downshift error @ pos {} (decode): {:?} — poisoning model",
                            m.seq_pos, e
                        );
                        crate::dense::emit_active_attempt_error(
                            stdout,
                            Some(id),
                            &format!("adaptive KV transition failed during decode: {e}"),
                            "transient",
                            true,
                            false,
                        );
                        let _ = stdout.flush();
                        return;
                    }
                }
            }

            if next_token == config.eos_token {
                break;
            }
            if im_end_token == Some(next_token) {
                break;
            }
            if tokenizer.is_terminator(next_token) {
                break;
            }

            // hunt3 M-F: user stop-sequence match against the decoded output
            // suffix. Matching on the full decoded text (not per-token) handles
            // stop strings that span a token boundary. On a hit we break out of
            // the decode loop; finish_reason naturally resolves to "stop" below
            // (hit_length_cap is false and no tool_calls were emitted).
            if !stop.is_empty() {
                let decoded_suffix = tokenizer.decode(&streamed_tokens);
                if stop.iter().any(|s| decoded_suffix.ends_with(s.as_str())) {
                    break;
                }
            }

            // max_think_tokens enforcement. Track whether we're inside an
            // open <think>...</think> block and how many tokens we've
            // emitted there. When the cap is hit, splice "</think>\n" into
            // the stream (KV write + stdout emit + advance generated) so
            // the model commits to an answer with the remaining budget.
            // Same decoded-text scan budget_alert uses; counter is
            // incremented per-iteration only when we're still inside.
            // Force-close the <think> span when EITHER the max_think_tokens
            // budget is hit OR the CLI sent a `force_answer` signal (a turn
            // running long → make the model commit to its answer instead of
            // the client timing out mid-think and terminating the stream).
            let force_answer_now = check_force_answer(id);
            // Latch: the CLI's force_answer is one-shot, so remember it for the
            // rest of the turn to keep enforcing the commit on any <think> re-open.
            if force_answer_now {
                force_answer_latched = true;
            }
            if max_think_tokens > 0
                || force_answer_now
                || force_answer_latched
                || max_total_think > 0
            {
                let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let in_think = currently_in_think(raw_str, started_in_think);
                // Total-think bound (re-arm-proof). Count every think token; at the
                // cap, latch force-answer (force-close + block <think>); a margin
                // past the cap, hard-EOS so a model that keeps re-opening <think>
                // can't run the turn out to the client timeout.
                if in_think {
                    total_think_tokens += 1;
                }
                if max_total_think > 0 && total_think_tokens >= max_total_think {
                    force_answer_latched = true;
                }
                if force_answer_latched && latch_gen_mark.is_none() {
                    latch_gen_mark = Some(generated);
                }
                if max_total_think > 0 && in_think && total_think_tokens >= max_total_think + 256 {
                    eprintln!("[think-cap] id={} — total think {} exceeded cap {}+256 while still thinking; forcing EOS", id, total_think_tokens, max_total_think);
                    break;
                }
                if let Some(mark) = latch_gen_mark {
                    if generated.saturating_sub(mark) >= post_latch_answer_budget {
                        eprintln!("[think-cap] id={} — {} tokens since think-cap latch without finishing; forcing EOS", id, generated.saturating_sub(mark));
                        break;
                    }
                }
                if max_think_tokens > 0 {
                    if in_think {
                        if !prev_in_think {
                            think_count = 1;
                        } else {
                            think_count += 1;
                        }
                    } else {
                        think_count = 0;
                    }
                    prev_in_think = in_think;
                }
                let budget_hit = max_think_tokens > 0 && think_count >= max_think_tokens;

                if in_think && (budget_hit || force_answer_now || force_answer_latched) {
                    if force_answer_now {
                        eprintln!("[force-answer] id={} — closing <think> mid-turn to commit to the answer", id);
                    } else if force_answer_latched {
                        eprintln!("[force-answer] id={} — re-closing a re-opened <think> (latched / think-cap)", id);
                    }
                    // Force-close. Encode the continuation and run each token
                    // through the KV write + emit path the same way a normally-
                    // sampled token does, so the model's next sample is
                    // conditioned on having "said" it (no hidden-state
                    // discontinuity). Respect max_tokens — clip if not enough
                    // room remains and bail.
                    let close_tokens = tokenizer.encode(&think_continuation());
                    let budget_left = max_tokens.saturating_sub(generated);
                    let take = close_tokens.len().min(budget_left);
                    for &t in &close_tokens[..take] {
                        // KV write before any emit — same contract as the main
                        // decode step (no token whose trunk write failed).
                        if let Err(e) = qwen35::forward_scratch(
                            gpu, weights, config, t, m.seq_pos, kv, dn, scratch,
                        ) {
                            let action = qwen_ar_forward_fail_action();
                            debug_assert!(!action.emit_failed_token);
                            if action.reset_uncommitted_state {
                                reset_ar_uncommitted_state!();
                            }
                            if action.emit_request_error {
                                write_error(
                                    stdout,
                                    id,
                                    &qwen_ar_forward_fail_message("forward_scratch think_close", e),
                                );
                            }
                            return;
                        }
                        // Keep the grammar matcher in sync over force-closed tokens,
                        // exactly as the normal sample path does. Without this, a
                        // tools request that force-closes <think> leaves the matcher
                        // in a stale state -> malformed/unparseable tool calls after
                        // the forced close.
                        if grammar_active {
                            grammar_matcher.advance(&tokenizer.decode(&[t]));
                        }
                        let prev_fed = bytes_fed_to_filter;
                        let elapsed_ms = t0.elapsed().as_millis() as u64;
                        match semantic.commit_and_classify(
                            stdout,
                            t,
                            || {
                                let pos = qwen_ar_raw_commit_token(
                                    &mut m.conversation_tokens,
                                    &mut streamed_tokens,
                                    &mut m.seq_pos,
                                    t,
                                    QwenArRawCommitDisposition::ClassifiedVisible,
                                );
                                if let Some(ref ev) = m.eviction {
                                    if let Some(hipfire_runtime::triattn::EvictionResult {
                                        new_physical: new_phys,
                                        ..
                                    }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap()
                                    {
                                        m.seq_pos = new_phys;
                                    }
                                }
                                let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
                                let new_bytes = all_bytes[prev_fed..].to_vec();
                                bytes_fed_to_filter = all_bytes.len();
                                (pos, new_bytes)
                            },
                            |pos, out| {
                                crate::common::emit_committed_event(out, id, t, pos, elapsed_ms);
                            },
                        ) {
                            Ok(true) => {
                                generated += 1;
                                break;
                            }
                            Ok(false) => {}
                            Err(err) => {
                                emit_error_with_id(stdout, id, &err.to_string());
                                return;
                            }
                        }
                        generated += 1;
                    }
                    think_count = 0;
                    prev_in_think = false;
                    if generated >= max_tokens {
                        break;
                    }
                }
            }

            // N-gram loop detector: check if any 4-gram in the recent window
            // repeats excessively. When detected, emit an info message and
            // force EOS to prevent wasting the remaining token budget on
            // repetitive output. Logic lives in `hipfire_runtime::loop_guard`.
            if let Some(hipfire_runtime::loop_guard::StopReason::NgramRepeat { count, .. }) =
                loop_guard.check(&streamed_tokens)
            {
                let window_len = loop_guard.window_len(streamed_tokens.len());
                emit_qwen_ar_info(
                    stdout,
                    id,
                    &format!(
                        "ngram loop detected (4gram repeated {}× in last {} tokens) — forcing EOS",
                        count, window_len
                    ),
                );
                break;
            }

            // Budget-alert injection: once we hit the configured token count,
            // splice the nudge text into the stream. Tokens are emitted to
            // stdout (so the client sees them) AND forward-fed through the KV
            // cache (so the model's next sample is conditioned on having
            // "said" them itself). Injected tokens count against `max_tokens`
            // — we never exceed the caller's requested budget — so we clip
            // the nudge if not enough room remains, and break out of the
            // outer loop if the budget is fully spent after injection.
            if !alert_fired
                && budget_alert_at_tok > 0
                && generated >= budget_alert_at_tok
                && !budget_alert_text.is_empty()
            {
                alert_fired = true;
                // Only inject while the model is inside an open <think> block.
                // The whole point of the feature is to nudge the model's
                // reasoning; firing past </think> just graffities the visible
                // answer with a system-alert string. Check the raw decoded
                // text rather than token IDs since <think> tokenizes as a
                // multi-token sequence in Qwen3.5's vocab.
                let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let in_think = currently_in_think(raw_str, started_in_think);
                if !in_think {
                    emit_qwen_ar_info(
                        stdout,
                        id,
                        "budget_alert skipped: not inside an open <think> block",
                    );
                    // Fall through — resample next token as normal
                    let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
                    let mut blocked: Vec<u32> = Vec::new();
                    sampler::collect_unclosed_attractor_blocks(
                        ngram_scope,
                        &attractor_pairs,
                        20,
                        2,
                        &mut blocked,
                    );
                    let cfg = SamplerConfig {
                        temperature: temp,
                        top_p,
                        repeat_penalty,
                        repeat_window: repeat_buf_cap,
                        presence_penalty,
                        frequency_penalty,
                        blocked_tokens: blocked,
                        top_k,
                        min_p,
                    };
                    next_token = if grammar_active && !grammar_matcher.is_free() {
                        let mut logits = gpu
                            .download_f32(&scratch.logits)
                            .unwrap_or_else(|_| vec![0.0f32; vocab_size]);
                        grammar_matcher.token_mask(grammar_vocab, &mut grammar_mask);
                        saddle_core::grammar::json::Matcher::apply_mask_to_logits(
                            &grammar_mask,
                            &mut logits,
                        );
                        sampler::sample_cpu(&mut logits, ngram_scope, &cfg)
                    } else {
                        sampler::sample(
                            gpu,
                            &scratch.logits,
                            &scratch.sample_buf,
                            &scratch.repeat_buf,
                            vocab_size,
                            ngram_scope,
                            &cfg,
                            &mut rng_state,
                        )
                    };
                    if grammar_active {
                        let text = tokenizer.decode(&[next_token]);
                        grammar_matcher.advance(&text);
                    }
                    continue;
                }
                let nudge_tokens = tokenizer.encode(budget_alert_text);
                let budget_left = max_tokens.saturating_sub(generated);
                let nudge_len = nudge_tokens.len().min(budget_left);
                // KV headroom check — don't run past physical_cap. If we don't
                // have room for the clipped nudge, skip entirely rather than
                // emit a partial nudge that poisons the trajectory. Under
                // eviction the physical check is trivially satisfied (budget
                // always holds post-evict), but we still respect the check for
                // the non-eviction path.
                let need_kv = m
                    .seq_pos
                    .saturating_add(nudge_len)
                    .saturating_add(
                        max_tokens
                            .saturating_sub(generated)
                            .saturating_sub(nudge_len),
                    )
                    .saturating_add(nl.len());
                if nudge_len > 0 && (m.eviction.is_some() || need_kv <= m.physical_cap) {
                    for &tok in &nudge_tokens[..nudge_len] {
                        // KV before emit: budget-alert injection must not stream
                        // a token if trunk VMM/forward fails mid-nudge.
                        if let Err(e) = qwen35::forward_scratch(
                            gpu, weights, config, tok, m.seq_pos, kv, dn, scratch,
                        ) {
                            let action = qwen_ar_forward_fail_action();
                            debug_assert!(!action.emit_failed_token);
                            if action.reset_uncommitted_state {
                                reset_ar_uncommitted_state!();
                            }
                            if action.emit_request_error {
                                write_error(
                                    stdout,
                                    id,
                                    &qwen_ar_forward_fail_message(
                                        "forward_scratch budget_alert",
                                        e,
                                    ),
                                );
                            }
                            return;
                        }
                        let prev_fed = bytes_fed_to_filter;
                        let elapsed_ms = t0.elapsed().as_millis() as u64;
                        match semantic.commit_and_classify(
                            stdout,
                            tok,
                            || {
                                let pos = qwen_ar_raw_commit_token(
                                    &mut m.conversation_tokens,
                                    &mut streamed_tokens,
                                    &mut m.seq_pos,
                                    tok,
                                    QwenArRawCommitDisposition::ClassifiedVisible,
                                );
                                if let Some(ref ev) = m.eviction {
                                    if let Some(hipfire_runtime::triattn::EvictionResult {
                                        new_physical: new_phys,
                                        ..
                                    }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap()
                                    {
                                        m.seq_pos = new_phys;
                                    }
                                }
                                // Injected text still goes through producer filter/router
                                // so think markers never leak on the contract-v2 wire.
                                let all_bytes2 = tokenizer.decode_bytes(&streamed_tokens);
                                let new_bytes2 = all_bytes2[prev_fed..].to_vec();
                                bytes_fed_to_filter = all_bytes2.len();
                                (pos, new_bytes2)
                            },
                            |pos, out| {
                                crate::common::emit_committed_event(out, id, tok, pos, elapsed_ms);
                            },
                        ) {
                            Ok(true) => {
                                generated += 1;
                                break;
                            }
                            Ok(false) => {}
                            Err(err) => {
                                emit_error_with_id(stdout, id, &err.to_string());
                                return;
                            }
                        }
                        generated += 1;
                    }
                } else if nudge_len < nudge_tokens.len() {
                    emit_qwen_ar_info(
                        stdout,
                        id,
                        &format!(
                            "budget_alert clipped or skipped: nudge_len={} budget_left={}",
                            nudge_len, budget_left
                        ),
                    );
                } else {
                    emit_qwen_ar_info(stdout, id, "budget_alert skipped: not enough KV headroom");
                }
                // Respect max_tokens: if injection used the remainder, bail
                // before sampling another model token.
                if generated >= max_tokens {
                    break;
                }
            }

            // Decide which paired-opener tokens (if any) trip the depth
            // threshold over a 20-token window. #111 attractor block —
            // cheap when not tripped, ~5 µs per blocked token when
            // tripped (single 4-byte H2D into the logits buffer
            // performed inside sampler::sample).
            let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
            let mut blocked: Vec<u32> = Vec::new();
            sampler::collect_unclosed_attractor_blocks(
                ngram_scope,
                &attractor_pairs,
                20,
                2,
                &mut blocked,
            );
            // Once force-answer has latched, forbid re-opening <think> so the
            // model commits to its answer instead of thinking unbounded.
            if force_answer_latched {
                if let Some(t) = think_open_tok {
                    blocked.push(t);
                }
            }
            let cfg = SamplerConfig {
                temperature: temp,
                top_p,
                repeat_penalty,
                repeat_window: repeat_buf_cap,
                presence_penalty,
                frequency_penalty,
                blocked_tokens: blocked,
                top_k,
                min_p,
            };
            // Grammar-gated sample (see setup block + tok0 site above).
            // GPU sample is the fast path; CPU mask-then-sample is the
            // constrained slow path that prevents the Pi turn-12
            // ChatML-noise-in-tool_call-body attractor.
            next_token = if grammar_active && !grammar_matcher.is_free() {
                let mut logits = gpu
                    .download_f32(&scratch.logits)
                    .unwrap_or_else(|_| vec![0.0f32; vocab_size]);
                grammar_matcher.token_mask(grammar_vocab, &mut grammar_mask);
                saddle_core::grammar::json::Matcher::apply_mask_to_logits(
                    &grammar_mask,
                    &mut logits,
                );
                sampler::sample_cpu(&mut logits, ngram_scope, &cfg)
            } else {
                sampler::sample(
                    gpu,
                    &scratch.logits,
                    &scratch.sample_buf,
                    &scratch.repeat_buf,
                    vocab_size,
                    ngram_scope,
                    &cfg,
                    &mut rng_state,
                )
            };
            if grammar_active {
                let text = tokenizer.decode(&[next_token]);
                let was_detected = grammar_matcher.attractor_detected();
                grammar_matcher.advance(&text);
                if !was_detected && grammar_matcher.attractor_detected() {
                    eprintln!(
                        "[grammar-ngram] attractor detected in tool_call args at gen={} — forcing close",
                        generated,
                    );
                }
            }
        }
        // m.seq_pos is already the "next physical write slot" — advanced
        // per-token in the decode loop above, and evicted back down to
        // `budget` whenever maybe_evict fired. No post-loop fix-up needed.

        // ChatML requires \n after <|im_end|>. Run it through forward so KV cache
        // and DeltaNet state stay in sync with seq_pos. Trailer tokens use the same
        // producer-owned raw-commit entry as decode tokens, with intentionally-hidden
        // disposition so they never join streamed/client-visible classify.
        if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
            for &t in &nl {
                if let Err(e) =
                    qwen35::forward_scratch(gpu, weights, config, t, m.seq_pos, kv, dn, scratch)
                {
                    let action = qwen_ar_forward_fail_action();
                    if action.reset_uncommitted_state {
                        reset_ar_uncommitted_state!();
                    }
                    if action.emit_request_error {
                        write_error(
                            stdout,
                            id,
                            &qwen_ar_forward_fail_message("forward_scratch trailer", e),
                        );
                    }
                    return;
                }
                // Producer-owned hidden commit: physical raw mutation + bookkeeping
                // once; no classify / no client-visible emit.
                let _ = semantic
                    .commit_raw(
                        stdout,
                        t,
                        QwenArRawCommitDisposition::IntentionallyHidden,
                        || {
                            let pos = qwen_ar_raw_commit_token(
                                &mut m.conversation_tokens,
                                &mut streamed_tokens,
                                &mut m.seq_pos,
                                t,
                                QwenArRawCommitDisposition::IntentionallyHidden,
                            );
                            (pos, Vec::<u8>::new())
                        },
                        |_pos, _out| {},
                    )
                    .expect("hidden commit never classifies");
                if let Some(ref ev) = m.eviction {
                    if let Some(hipfire_runtime::triattn::EvictionResult {
                        new_physical: new_phys,
                        ..
                    }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap()
                    {
                        m.seq_pos = new_phys;
                    }
                }
            }
        }

        // ── semantic finish: tool_calls + cache + done ─────────────
        //
        // QwenArSemanticProducer is the sole wire authority for this AR path.
        // Raw streamed_tokens still hold full protocol bytes for KV replay;
        // client tokens were already stripped of markers during the loop.
        // Length-cap never exposes executable calls; malformed without
        // length fails closed (error, no tool_calls, no asst_turn_cache).
        // Decoded EOT on the final budget token beats length.
        // Open think → validation terminal (no cache) via finish().
        // finish() shares the same drain + classify path as unit tests.
        let hit_length_cap = generated >= max_tokens;
        let (finish, visible_for_cache) = match semantic.finish(stdout, hit_length_cap) {
            Ok(pair) => pair,
            Err(err) => {
                emit_error_with_id(stdout, id, &err.to_string());
                return;
            }
        };
        // Open-think: production owns epilogue + single correlated error terminal.
        if matches!(finish.cause, QwenArTerminalCause::OpenThink) {
            let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
            emit_qwen_ar_open_think_terminal(stdout, id, generated, &ep);
            return;
        }
        //
        // Timing + pending done are fixed before handshake so commit_ready
        // carries the exact eventual done payload. Abort rolls back + emits
        // one cancellation lifecycle (or fail-closed error if unattested).
        let t_end = Instant::now();
        let total_s = t_end.duration_since(t0).as_secs_f64();
        let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
        let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
        let tok_s = if total_s > 0.0 {
            generated as f64 / total_s
        } else {
            0.0
        };
        let prefill_tok_s = if prefill_s > 0.0 {
            prefill_tokens as f64 / prefill_s
        } else {
            0.0
        };
        let decode_tok_s = if decode_s > 0.0 {
            generated as f64 / decode_s
        } else {
            0.0
        };
        let pflash_frag =
            pflash_done_fragment(&pflash_summary, &pflash_bypass_reason, pflash_alpha);
        let mut pending_done = qwen_ar_done_value(
            id,
            finish.finish_reason,
            generated,
            tok_s,
            prefill_tokens,
            prefill_s * 1000.0,
            prefill_tok_s,
            decode_tok_s,
            prefill_s * 1000.0,
            cached_tokens_count,
            &pflash_frag,
        );
        // Stage canonical calls in commit_ready/done for tool-safe terminals.
        // No post-commit tool_calls event — engine drain sees only final done.
        stage_terminal_tool_calls(
            &mut pending_done,
            finish.finish_reason,
            &finish.wire_tool_calls,
        );
        let decision = await_client_terminal_commit(stdout, id, &pending_done);
        let intended_release =
            finish.finish_reason == "tool_calls" && !finish.wire_tool_calls.is_empty();
        let intended_store = finish.store_cache;
        let effects =
            crate::qwen::qwen_client_commit_effects(decision, intended_release, intended_store);
        if !effects.emit_done {
            let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
            crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
            return;
        }
        // Tool release side effects (cache fingerprint uses wire_tool_calls)
        // remain gated on Commit; wire delivery is via staged terminal only.
        //
        // Store the model's verbatim emitted token sequence under a
        // fingerprint over (visible stripped text, wire tool_calls) so the
        // next turn's prompt-cache renderer can replay the exact bytes
        // the model wrote into KV instead of re-encoding via
        // `tokenizer.encode(msg.content)` (BPE non-bijective).
        //
        // Suppressed on malformed / length-partial / open-think turns and
        // on client Abort (effects.store_cache == false).
        let mut cache_action = qwen_ar_cache_action(&finish, &visible_for_cache);
        cache_action.store = effects.store_cache && cache_action.store;
        if cache_action.store {
            let mut cached_seq: Vec<u32> =
                m.conversation_tokens[decode_start_tokens_idx..].to_vec();
            // Trim trailing `\n` newline tokens from the forced trailer.
            while let Some(&last) = cached_seq.last() {
                if nl.contains(&last) {
                    cached_seq.pop();
                } else {
                    break;
                }
            }
            // Trim a single trailing `<|im_end|>` (if the tokenizer
            // registered it as one token id).
            if let Some(&last) = cached_seq.last() {
                if im_end_token == Some(last) {
                    cached_seq.pop();
                }
            }
            if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
                eprintln!(
                    "[qwen-cache store] cached_seq={} emit_text.len={} tool_calls={} preview={:?}",
                    cached_seq.len(),
                    cache_action.fingerprint_text.len(),
                    cache_action.tool_calls.len(),
                    cache_action
                        .fingerprint_text
                        .chars()
                        .take(60)
                        .collect::<String>(),
                );
            }
            let _ = qwen_ar_apply_cache_action(
                |fp, seq| {
                    m.asst_turn_cache.insert(
                        fp,
                        hipfire_runtime::prompt_frame::CachedAssistantTurn {
                            reasoning: None,
                            tools: Vec::new(),
                            content: Some(hipfire_runtime::prompt_frame::CachedAssistantBody {
                                token_ids: seq,
                                text: String::new(),
                            }),
                        },
                    )
                },
                &cache_action,
                cached_seq,
            );
        }

        emit_staged_terminal_done(stdout, &pending_done);
    } else {
        // LLaMA path -- multi-turn aware
        let has_eviction = m.eviction.is_some();
        let b = m
            .state
            .as_mut()
            .and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_llama::LlamaBundle>()
            })
            .unwrap();
        let config = &b.config;
        let weights = &b.weights;
        let scratch = &b.scratch;
        let kv = &mut b.kv;

        let mut rng_state = 42u32;
        let batched_prefill = llama_qwen3_batched_prefill_eligible(
            &gpu.arch,
            config.arch,
            hipfire_runtime::config::get().prefill_batched,
            kv.quant_q8,
            has_eviction,
            new_tokens.len(),
        );
        let (mut next_token, sampled_rng) = if batched_prefill {
            llama::forward_prefill_batch(
                gpu,
                weights,
                config,
                &new_tokens,
                m.seq_pos,
                kv,
                scratch,
                None,
            )
            .unwrap();
            let sample_seed = llama_prefill_sample_seed(rng_state, new_tokens.len(), temp);
            gpu.sample_top_p(
                &scratch.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                config.vocab_size,
                temp,
                top_p,
                sample_seed,
                0,
                1.0,
            )
            .unwrap()
        } else {
            for (i, &tok) in new_tokens.iter().enumerate() {
                let pos = m.seq_pos + i;
                let (_, rng) = llama::forward_scratch(
                    gpu, weights, config, tok, pos, kv, scratch, temp, top_p, rng_state, 0, 1.0,
                )
                .unwrap();
                rng_state = rng;
            }
            let mut out_bytes = [0u8; 8];
            gpu.hip
                .memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf)
                .unwrap();
            (
                u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]),
                u32::from_ne_bytes([out_bytes[4], out_bytes[5], out_bytes[6], out_bytes[7]]),
            )
        };
        rng_state = sampled_rng;
        let this_turn_prompt_len_llama = new_tokens.len();
        m.seq_pos += new_tokens.len();
        m.conversation_tokens.extend_from_slice(&new_tokens);
        let ngram_scope_start_llama = m.conversation_tokens.len() - this_turn_prompt_len_llama;
        // Prefill ends here: prompt is processed AND first token is ready (D2H
        // sync is the user-observable "time to first token" boundary). Decode
        // below measures the pure forward+sample steady-state.
        let t_prefill = Instant::now();

        let mut generated = 0;
        let mut streamed_tokens: Vec<u32> = Vec::new();
        // `bytes_fed_to_filter` is the index into the freshly-decoded
        // byte stream past which we have not yet handed bytes to the
        // filter. The filter owns UTF-8 boundary buffering and any
        // future arch quirks (Gemma 4 marker holdback, strip-think,
        // byte-level stop_at); see crates/engine/src/eos_filter.rs.
        let mut bytes_fed_to_filter = 0usize;
        let mut filter = EosFilter::new(EosFilterConfig::default());

        for _ in 0..max_tokens {
            generated += 1;
            m.conversation_tokens.push(next_token);
            streamed_tokens.push(next_token);
            crate::common::emit_committed_event(
                stdout,
                id,
                next_token,
                streamed_tokens.len() - 1,
                t0.elapsed().as_millis() as u64,
            );
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[bytes_fed_to_filter..];
            bytes_fed_to_filter = all_bytes.len();
            if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                let text = std::str::from_utf8(&text_bytes).unwrap();
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                    id,
                    serde_json::to_string(&text).unwrap_or_default(),
                    active_attempt_id()
                );
                let _ = stdout.flush();
            }

            // Scope repeat_buf to this turn's prompt + generated tokens
            // (same logic as the Qwen3.5 path: prompt anchor + current turn).
            let rw = repeat_window.min(64);
            let scope_start =
                ngram_scope_start_llama.max(m.conversation_tokens.len().saturating_sub(rw));
            let hist_slice = &m.conversation_tokens[scope_start..];
            let hist_bytes: Vec<u8> = hist_slice.iter().flat_map(|t| t.to_ne_bytes()).collect();
            gpu.hip
                .memcpy_htod(&scratch.repeat_buf.buf, &hist_bytes)
                .unwrap();

            // Write K/V for this token FIRST so the next turn's context is
            // always fully populated. The sampled next_token from this call
            // is discarded when we break on im_end/eos — wasteful by one
            // launch but avoids a KV cache gap at the terminator.
            let pos = m.seq_pos + generated - 1;
            let (tok, rng) = llama::forward_scratch(
                gpu,
                weights,
                config,
                next_token,
                pos,
                kv,
                scratch,
                temp,
                top_p,
                rng_state,
                hist_slice.len(),
                repeat_penalty,
            )
            .unwrap();

            if next_token == config.eos_token {
                break;
            }
            if im_end_token == Some(next_token) {
                break;
            }
            if tokenizer.is_terminator(next_token) {
                break;
            }

            next_token = tok;
            rng_state = rng;
        }
        m.seq_pos += generated;

        // ChatML \n boundary — run through forward to keep KV cache in sync
        if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
            for &t in &nl {
                let (_, rng2) = llama::forward_scratch(
                    gpu, weights, config, t, m.seq_pos, kv, scratch, temp, top_p, rng_state, 0, 1.0,
                )
                .unwrap();
                rng_state = rng2;
                m.seq_pos += 1;
                m.conversation_tokens.push(t);
            }
        }

        let t_end = Instant::now();
        let total_s = t_end.duration_since(t0).as_secs_f64();
        let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
        let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
        let tok_s = if total_s > 0.0 {
            generated as f64 / total_s
        } else {
            0.0
        };
        let prefill_tok_s = if prefill_s > 0.0 {
            prefill_tokens as f64 / prefill_s
        } else {
            0.0
        };
        let decode_tok_s = if decode_s > 0.0 {
            generated as f64 / decode_s
        } else {
            0.0
        };
        let pflash_frag =
            pflash_done_fragment(&pflash_summary, &pflash_bypass_reason, pflash_alpha);
        let mut pending_done = serde_json::json!({
            "type": "done",
            "id": id,
            "tokens": generated,
            "tok_s": (tok_s * 10.0).round() / 10.0,
            "prefill_tokens": prefill_tokens,
            "prefill_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
            "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
            "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
            "ttft_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
            "attempt_id": active_attempt_id(),
        });
        if !pflash_frag.is_empty() {
            let padded = format!("{{{}}}", pflash_frag.trim_start_matches(','));
            if let Ok(serde_json::Value::Object(map)) =
                serde_json::from_str::<serde_json::Value>(&padded)
            {
                for (k, v) in map {
                    pending_done[k] = v;
                }
            }
        }
        match await_client_terminal_commit(stdout, id, &pending_done) {
            ClientTerminalDecision::Commit => emit_staged_terminal_done(stdout, &pending_done),
            ClientTerminalDecision::Abort => {
                // Bring-up AR path has no full production rollback attestation;
                // suppress success done on cancel/disconnect (fail-closed).
            }
        }
    }
}

/// Resolve whether DeepSeek4 spec-decode is requested from the installed
/// drafter and typed MTP policy.
///
/// DSpark is a distinct speculation selector, not an MTP mode.  The loader has
/// already applied the typed `dspark_mode` policy before installing the
/// speculator, so an installed DSpark drafter is authoritative here.  Falling
/// through to `mtp_mode` for DSpark made `speculation = "dspark"` load and build
/// the sidecar, then silently route the request through AR because that selector
/// correctly pins MTP off.
pub fn deepseek4_spec_requested_from_policy(
    drafter_name: Option<&str>,
    process_mtp_mode: &str,
    model_mtp_mode: &str,
    model_mtp_weights_present: bool,
) -> bool {
    if drafter_name == Some("dspark") {
        return true;
    }
    match process_mtp_mode {
        "on" => true,
        "off" => false,
        _ => model_mtp_mode == "on" || (model_mtp_mode == "auto" && model_mtp_weights_present),
    }
}

pub fn deepseek4_spec_requested(m: &LoadedModel) -> bool {
    deepseek4_spec_requested_from_policy(
        m.speculator.as_ref().map(|s| s.name()),
        hipfire_runtime::config::get().mtp_mode.as_str(),
        &m.mtp_mode,
        m.mtp_weights_present,
    )
}

/// Emit the full Qwen AR `done` envelope via serde (hostile-id safe).
pub fn emit_qwen_ar_done(
    stdout: &mut impl std::io::Write,
    id: &str,
    finish_reason: &str,
    generated: usize,
    tok_s: f64,
    prefill_tokens: usize,
    prefill_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    ttft_ms: f64,
    cached_tokens: usize,
    pflash_fragment_json: &str,
) {
    let envelope = qwen_ar_done_value(
        id,
        finish_reason,
        generated,
        tok_s,
        prefill_tokens,
        prefill_ms,
        prefill_tok_s,
        decode_tok_s,
        ttft_ms,
        cached_tokens,
        pflash_fragment_json,
    );
    emit_staged_terminal_done(stdout, &envelope);
}

pub fn model_retry_reset_eligible(arch_id: u32) -> bool {
    hipfire_runtime::reset_core::is_retry_reset_eligible(reset_core_arch_key(arch_id))
}

/// Map LoadedModel.arch_id to reset_core inventory arch key.
pub fn reset_core_arch_key(arch_id: u32) -> &'static str {
    match arch_id {
        0 | 1 => "llama",
        5 | 6 => "qwen35",
        7 => "qwen2",
        8 => "dots-ocr",
        9 => "deepseek4",
        10 => "minimax",
        11 => "lfm2moe",
        12 => "cohere2moe",
        13 => "gemma4",
        14 => "muse_glimmer",
        _ => "unknown",
    }
}
