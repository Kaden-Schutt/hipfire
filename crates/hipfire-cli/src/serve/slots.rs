// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Multi-slot concurrent engine.
//!
//! Concern: concurrent in-process `SlotEngine` backend, its tokenizer, and
//! the `complete_request_slots` fast path. Behind the `multi-slot` Cargo feature;
//! isolates the only arch-specific crate (`hipfire-arch-qwen35`) so the CLI
//! can be built arch-free with `--no-default-features`.

#[cfg(feature = "multi-slot")]
use anyhow::{anyhow, bail, Context, Result};
#[cfg(not(feature = "multi-slot"))]
use anyhow::{bail, Result};
use std::sync::{Arc, mpsc};
#[cfg(feature = "multi-slot")]
use hipfire_runtime::tokenizer::Tokenizer;
#[cfg(feature = "multi-slot")]
use hipfire_runtime::prompt_frame::{AssistantPrefix, ChatFrame, Role};
#[cfg(feature = "multi-slot")]
use hipfire_runtime::serve::{Event, SubmitRequest};
use serde_json;
use crate::serve::complete::Completion;

#[cfg(feature = "multi-slot")]
pub(crate) struct SlotBackend {
    pub(crate) engine: hipfire_arch_qwen35::serve_engine::SlotEngine,
    pub(crate) tokenizer: hipfire_runtime::tokenizer::Tokenizer,
}

#[cfg(not(feature = "multi-slot"))]
pub(crate) struct SlotBackend;


#[cfg(feature = "multi-slot")]
pub(crate) fn complete_request_slots(
    backend: &SlotBackend,
    body: &serde_json::Value,
    identity: &(String, u64),
    event_callback: &mut dyn FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
    terminal_callback: &mut dyn FnMut(&Completion) -> Result<(), hipfire_client::ClientError>,
) -> Result<Completion> {
    use hipfire_runtime::emit_text::{ThinkOutputRouter, ThinkRouteEvent};
    use hipfire_runtime::prompt_frame::{AssistantPrefix, ChatFrame, Role};
    use hipfire_runtime::serve::{Event, SubmitRequest};

    let model = body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    let max_tokens = body
        .get("max_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(512) as usize;

    // Messages -> the shape ChatFrame actually wants. This is not a free
    // mapping: `build_multi_turn` PANICS on a System or Tool role inside the
    // history, because system text belongs in `ChatFrame.system` and the final
    // user turn in `ChatFrame.user`. History carries only the prior
    // User/Assistant exchange.
    let mut system: Option<String> = None;
    let mut turns: Vec<(Role, String)> = Vec::new();
    let Some(msgs) = body.get("messages").and_then(serde_json::Value::as_array) else {
        bail!("messages is required");
    };
    for m in msgs {
        let text = m
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned();
        match m.get("role").and_then(serde_json::Value::as_str) {
            Some("system") => {
                // Multiple system messages concatenate, matching how the
                // single-session path folds them.
                match system.as_mut() {
                    Some(existing) => {
                        existing.push('\n');
                        existing.push_str(&text);
                    }
                    None => system = Some(text),
                }
            }
            Some("assistant") => turns.push((Role::Assistant, text)),
            // A tool result reads as user-side context to the model.
            Some("tool") => turns.push((Role::User, text)),
            _ => turns.push((Role::User, text)),
        }
    }
    // The final user turn is the prompt; everything before it is history.
    let last_user = match turns.iter().rposition(|(r, _)| *r == Role::User) {
        Some(i) => turns.remove(i).1,
        None => bail!("messages must contain at least one user message"),
    };
    let history: Vec<(Role, &str)> = turns.iter().map(|(r, t)| (*r, t.as_str())).collect();
    // A thinking model must be handed an OPEN <think> block, as the daemon's
    // spec_assistant_prefix does. With Plain, the model has to emit `<think>`
    // itself and instead loops on `</think>` and re-opens user turns.
    let prefix = if backend.tokenizer.special_token_id("<think>").is_some() {
        AssistantPrefix::OpenThink
    } else {
        AssistantPrefix::Plain
    };
    let frame = ChatFrame {
        tokenizer: &backend.tokenizer,
        system: system.as_deref(),
        user: &last_user,
        assistant_prefix: prefix,
        raw: false,
    };
    let prompt_tokens = frame.build_multi_turn(&history);

    // Conversation identity is the USER turns. The assistant side is whatever
    // we generated, and the client's echo of it may differ (reasoning split to
    // its own channel, whitespace, an edited message), so it cannot be part of
    // the key.
    pub(crate) fn turn_hash(s: &str) -> u64 {
        let mut h = 0xcbf29ce484222325_u64;
        for b in s.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
    let mut convo: Vec<u64> = turns
        .iter()
        .filter(|(r, _)| *r == Role::User)
        .map(|(_, t)| turn_hash(t))
        .collect();
    convo.push(turn_hash(&last_user));

    // Tokens that continue the previous assistant turn into this user turn.
    // The engine appends these to the session's exact stored tokens instead of
    // re-rendering the history, which is what makes the result a strict
    // extension of the KV. Only meaningful from turn 2 onwards.
    let continuation = if convo.len() >= 2 {
        hipfire_runtime::prompt_frame::continuation_suffix(&backend.tokenizer, &last_user, prefix)
    } else {
        Vec::new()
    };

    let (tx, rx) = mpsc::channel::<Event>();
    backend
        .engine
        .submit(SubmitRequest {
            session: None,
            prompt_tokens,
            convo,
            continuation,
            max_tokens,
            reply: tx,
        })
        .map_err(|e| anyhow!("multi_slot submit: {e}"))?;

    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut finish = "stop";

    // Split <think> spans out of the visible answer. Without this the whole
    // reasoning preamble lands in `content` -- a 4B answer arrives as ~2 KB of
    // "Thinking Process: 1. **Analyze the Request**..." with reasoning_content
    // empty. `started_in_think` is true because the prompt itself opened the
    // block (AssistantPrefix::OpenThink above), so generation begins inside it
    // and there is no opening marker in the generated stream to detect.
    let mut think = ThinkOutputRouter::new(matches!(prefix, AssistantPrefix::OpenThink));
    let mut routed: Vec<ThinkRouteEvent> = Vec::new();

    let mut drain_routed = |routed: &mut Vec<ThinkRouteEvent>,
                            content: &mut String,
                            reasoning_content: &mut String,
                            cb: &mut dyn FnMut(
        &serde_json::Value,
    ) -> Result<(), hipfire_client::ClientError>| {
        for ev in routed.drain(..) {
            match ev {
                ThinkRouteEvent::Content(text) => {
                    content.push_str(&text);
                    let _ = cb(&serde_json::json!({ "type": "token", "text": text }));
                }
                ThinkRouteEvent::Reasoning(text) => {
                    reasoning_content.push_str(&text);
                    let _ = cb(&serde_json::json!({ "type": "reasoning", "text": text }));
                }
            }
        }
    };

    while let Ok(ev) = rx.recv() {
        match ev {
            Event::Accepted { .. } => {}
            Event::Token { id } => {
                let text = backend.tokenizer.decode(&[id]);
                if !text.is_empty() {
                    think.push_into(&text, &mut routed);
                    drain_routed(
                        &mut routed,
                        &mut content,
                        &mut reasoning_content,
                        event_callback,
                    );
                }
            }
            Event::Done { reason } => {
                finish = match reason {
                    hipfire_runtime::serve::DoneReason::MaxTokens => "length",
                    _ => "stop",
                };
                break;
            }
            Event::Rejected { reason } => {
                // Saturation is a real, expected answer here, not a crash: all
                // slots busy means the caller should retry, not that anything
                // failed.
                bail!("multi_slot rejected: {reason}");
            }
        }
    }

    // Flush any trailing partial marker as ordinary text in its channel.
    think.finish_into(&mut routed);
    drain_routed(
        &mut routed,
        &mut content,
        &mut reasoning_content,
        event_callback,
    );

    let completion = Completion {
        id: identity.0.clone(),
        created: identity.1,
        model,
        content,
        reasoning_content,
        preserve_thinking: false,
        tool_calls: Vec::new(),
        done: serde_json::json!({ "finish_reason": finish }),
    };
    // The terminal callback is what stages the response body and signals the
    // HTTP handler that the request succeeded. Skipping it leaves the handler
    // waiting on a status that never arrives, which surfaces to the client as
    // "generation worker disconnected".
    terminal_callback(&completion).map_err(|e| anyhow!("terminal callback: {e}"))?;
    Ok(completion)
}


#[cfg(not(feature = "multi-slot"))]
pub(crate) fn complete_request_slots(
    _backend: &SlotBackend,
    _body: &serde_json::Value,
    _identity: &(String, u64),
    _event_callback: &mut dyn FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
    _terminal_callback: &mut dyn FnMut(&Completion) -> Result<(), hipfire_client::ClientError>,
) -> Result<Completion> {
    bail!("multi-slot feature disabled")
}