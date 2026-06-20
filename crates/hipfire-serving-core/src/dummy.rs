// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `hipfire:dummy` model backend — a GPU-free token/word counter used by
//! tests and the batch-prefill harness to exercise the daemon protocol without
//! loading real weights. Extracted verbatim from the former `main.rs` monolith
//! (no behavior change); items are `pub`.

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use hipfire_generate::{GenerateBatchPrefillEnvelope, GenerateBatchPrefillSession};

/// GPU-free stand-in model: per-session token counters keyed by session id. Used
/// by tests and the batch-prefill harness to drive the daemon protocol end to
/// end without loading weights. Emits synthetic `dummy:N` tokens.
#[derive(Default)]
pub struct DummyModelState {
    pub sessions: HashMap<String, usize>,
}

impl DummyModelState {
    /// Drop all per-session counters (cold start).
    pub fn reset(&mut self) {
        self.sessions.clear();
    }

    /// Forget the given sessions; returns how many were actually present.
    pub fn release_sessions(&mut self, sessions: &[String]) -> usize {
        sessions
            .iter()
            .filter(|session| self.sessions.remove(*session).is_some())
            .count()
    }

    /// Number of resident sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Whitespace-word count — the dummy's stand-in for real tokenization.
    fn prompt_token_count(text: &str) -> usize {
        text.trim()
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .count()
    }

    /// Advance a session's counter by its prompt/suffix token count (seeding the
    /// counter from `logical_position` on first sight); returns tokens consumed.
    pub fn consume_prefill_session(&mut self, session: &GenerateBatchPrefillSession) -> usize {
        let consumed = match (&session.prompt, &session.suffix_tokens) {
            (Some(prompt), None) => {
                Self::prompt_token_count(prompt)
                    + session
                        .system_prompt
                        .as_deref()
                        .map(Self::prompt_token_count)
                        .unwrap_or(0)
            }
            (None, Some(tokens)) => tokens.len(),
            _ => 0,
        };
        let counter = self
            .sessions
            .entry(session.id.clone())
            .or_insert(session.state_handle.logical_position);
        *counter += consumed;
        consumed
    }

    /// Stream `max_tokens` synthetic `dummy:N` token events then a `done`
    /// envelope, mirroring the real generate protocol (with an optional
    /// configurable delay) so clients can be exercised GPU-free.
    pub fn generate(
        &mut self,
        stdout: &mut std::io::Stdout,
        id: &str,
        session_id: &str,
        prompt: &str,
        prefill_already_done: bool,
        max_tokens: usize,
    ) {
        let counter = self.sessions.entry(session_id.to_string()).or_insert(0);
        if !prefill_already_done {
            *counter += Self::prompt_token_count(prompt);
        }
        let generate_delay_ms = dummy_generate_delay_ms();
        if generate_delay_ms > 0 {
            tracing::debug!(
                request_id = id,
                session_id,
                delay_ms = generate_delay_ms,
                "dummy generate delay"
            );
            std::thread::sleep(std::time::Duration::from_millis(generate_delay_ms));
        }
        let started_at = Instant::now();
        for i in 0..max_tokens {
            let token = format!("dummy:{}", *counter);
            *counter += 1;
            let line = serde_json::json!({
                "type": "token",
                "id": id,
                "text": if i == 0 { token } else { format!(" {token}") },
            });
            let _ = writeln!(stdout, "{line}");
            let _ = stdout.flush();
        }
        let elapsed = started_at.elapsed().as_secs_f64().max(0.000_001);
        let done = serde_json::json!({
            "type": "done",
            "id": id,
            "tokens": max_tokens,
            "prefill_tokens": if prefill_already_done { 0 } else { Self::prompt_token_count(prompt) },
            "tok_s": (max_tokens as f64) / elapsed,
            "finish_reason": "length",
        });
        let _ = writeln!(stdout, "{done}");
        let _ = stdout.flush();
    }
}

/// Optional synthetic prefill delay (`HIPFIRE_DUMMY_PREFILL_DELAY_MS`, clamped
/// 0–5000) for exercising client timeout/latency handling.
fn dummy_prefill_delay_ms() -> u64 {
    std::env::var("HIPFIRE_DUMMY_PREFILL_DELAY_MS")
        .ok()
        .and_then(|raw| raw.parse::<i64>().ok())
        .unwrap_or(0)
        .clamp(0, 5000) as u64
}

/// Optional pseudo-random per-generate delay in `0..=max` ms, where `max` is
/// `HIPFIRE_DUMMY_GENERATE_DELAY_MS` (default 8, clamped 0–250). The jitter comes
/// from the wall clock, giving non-uniform timings for client stress tests.
fn dummy_generate_delay_ms() -> u64 {
    let max_ms = std::env::var("HIPFIRE_DUMMY_GENERATE_DELAY_MS")
        .ok()
        .and_then(|raw| raw.parse::<i64>().ok())
        .unwrap_or(8)
        .clamp(0, 250) as u64;
    if max_ms == 0 {
        return 0;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    nanos % (max_ms + 1)
}

/// Emit the `generate_batch_prefill_ready` capability envelope advertising the
/// dummy backend can serve a batch-prefill request.
pub fn emit_dummy_generate_batch_prefill_ready(
    stdout: &mut std::io::Stdout,
    envelope: &GenerateBatchPrefillEnvelope,
) {
    let line = serde_json::json!({
        "type": "generate_batch_prefill_ready",
        "id": envelope.id,
        "batch_id": envelope.batch_id,
        "sessions": envelope.session_count,
        "supported": true,
        "mode": "dummy_counter",
        "reason": "dummy_generate_batch_prefill_available",
        "target_model": "hipfire:dummy",
        "target_module": "dummy_prefill",
    });
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

/// Run a dummy batch prefill: emit `started`, consume each session's tokens into
/// its counter emitting a per-session `done`, then a batch `done` — the full
/// protocol shape the real backends produce, GPU-free.
pub fn run_generate_batch_prefill_dummy(
    dummy: &mut DummyModelState,
    stdout: &mut std::io::Stdout,
    envelope: &GenerateBatchPrefillEnvelope,
) -> Result<(), String> {
    let delay_ms = dummy_prefill_delay_ms();
    let started = serde_json::json!({
        "type": "generate_batch_prefill_started",
        "id": envelope.id,
        "batch_id": envelope.batch_id,
        "sessions": envelope.session_count,
        "mode": "dummy_counter",
        "plan": "dummy_counter",
        "backend": "dummy_delay",
        "delay_ms": delay_ms,
        "target_model": "hipfire:dummy",
        "target_module": "dummy_prefill",
        "state_kinds": ["attention_kv"],
    });
    let _ = writeln!(stdout, "{started}");
    let _ = stdout.flush();

    if delay_ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
    }

    let t0 = Instant::now();
    let mut total_prefill_tokens = 0usize;
    for session in &envelope.sessions {
        let consumed = dummy.consume_prefill_session(session);
        total_prefill_tokens += consumed;
        let logical_position = *dummy.sessions.get(&session.id).unwrap_or(&0);
        let line = serde_json::json!({
            "type": "generate_batch_prefill_session_done",
            "id": envelope.id,
            "batch_id": envelope.batch_id,
            "session_id": session.id,
            "prefill_tokens": consumed,
            "logical_position": logical_position,
            "cached_prefix_tokens": session.state_handle.cached_prefix_tokens,
            "state_handle": {
                "kind": "dummy_session",
                "runtime_state": "resident",
                "session_id": session.id,
                "logical_position": logical_position,
                "cached_prefix_tokens": session.state_handle.cached_prefix_tokens,
                "state_kinds": session.state_handle.state_kinds,
            },
            "mode": "dummy_counter",
            "plan": "dummy_counter",
            "backend": "dummy_delay",
            "delay_ms": delay_ms,
            "target_model": "hipfire:dummy",
            "target_module": "dummy_prefill",
            "state_kinds": session.state_handle.state_kinds,
        });
        let _ = writeln!(stdout, "{line}");
        let _ = stdout.flush();
    }

    let done = serde_json::json!({
        "type": "generate_batch_prefill_done",
        "id": envelope.id,
        "batch_id": envelope.batch_id,
        "sessions": envelope.session_count,
        "prefill_tokens": total_prefill_tokens,
        "elapsed_ms": t0.elapsed().as_secs_f64() * 1000.0 + delay_ms as f64,
        "mode": "dummy_counter",
        "plan": "dummy_counter",
        "backend": "dummy_delay",
        "resident_sessions": dummy.session_count(),
        "delay_ms": delay_ms,
        "target_model": "hipfire:dummy",
        "target_module": "dummy_prefill",
        "state_kinds": ["attention_kv"],
    });
    let _ = writeln!(stdout, "{done}");
    let _ = stdout.flush();
    Ok(())
}
