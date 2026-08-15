// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
        Arc,
    },
    time::Duration,
};

use anyhow::Result;
use hipfire_client::{stream_openai_chat, ClientError, OpenAiSseEvent};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug)]
pub enum ChatEvent {
    Reasoning(String),
    Content(String),
    Done,
    Error(String),
}

pub fn stream_chat(
    host: &str,
    port: u16,
    model: &str,
    messages: &[ChatMessage],
    temperature: Option<f64>,
    top_p: Option<f64>,
    tx: Sender<ChatEvent>,
    abort: Arc<AtomicBool>,
) -> Result<()> {
    let result = stream_chat_inner(host, port, model, messages, temperature, top_p, &tx, &abort);
    if let Err(err) = result {
        let _ = tx.send(ChatEvent::Error(err.to_string()));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn stream_chat_inner(
    host: &str,
    port: u16,
    model: &str,
    messages: &[ChatMessage],
    temperature: Option<f64>,
    top_p: Option<f64>,
    tx: &Sender<ChatEvent>,
    abort: &Arc<AtomicBool>,
) -> Result<()> {
    let mut body = json!({
        "model": model,
        "messages": messages,
    });
    // Per-session sampling overrides (set via /temp and /top_p).
    if let Some(t) = temperature {
        body["temperature"] = json!(t);
    }
    if let Some(p) = top_p {
        body["top_p"] = json!(p);
    }
    match stream_openai_chat(
        host,
        port,
        body,
        Duration::from_secs(600),
        |event| {
            match event {
                OpenAiSseEvent::Reasoning { text } => {
                    let _ = tx.send(ChatEvent::Reasoning(text));
                }
                OpenAiSseEvent::Content { text } => {
                    let _ = tx.send(ChatEvent::Content(text));
                }
                OpenAiSseEvent::Role { .. }
                | OpenAiSseEvent::ToolCall { .. }
                | OpenAiSseEvent::Finish { .. }
                | OpenAiSseEvent::Usage { .. }
                | OpenAiSseEvent::Done => {}
            }
            Ok(())
        },
        || abort.load(Ordering::Relaxed),
    ) {
        Ok(()) => {
            let _ = tx.send(ChatEvent::Done);
            Ok(())
        }
        // Explicit client cancel (Esc) is not an error and is never retried.
        Err(ClientError::Cancelled) => {
            let _ = tx.send(ChatEvent::Done);
            Ok(())
        }
        Err(err) => Err(err.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_reasoning_serialization() {
        // None is omitted; Some is emitted as reasoning_content.
        let msg_none = ChatMessage {
            role: "assistant".into(),
            content: "answer".into(),
            reasoning_content: None,
        };
        let v = serde_json::to_value(&msg_none).unwrap();
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"], "answer");
        assert!(v.get("reasoning_content").is_none(), "None must not serialize");

        let msg_some = ChatMessage {
            role: "assistant".into(),
            content: "answer".into(),
            reasoning_content: Some("think".into()),
        };
        let v = serde_json::to_value(&msg_some).unwrap();
        assert_eq!(v["reasoning_content"], "think");
        assert_eq!(v["content"], "answer");

        // Deserializing legacy JSON without the field yields None.
        let legacy: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": "hello"
        }))
        .unwrap();
        assert!(legacy.reasoning_content.is_none());

        // Deserializing with reasoning_content populates it.
        let with_reasoning: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": "hello",
            "reasoning_content": "plan"
        }))
        .unwrap();
        assert_eq!(with_reasoning.reasoning_content.as_deref(), Some("plan"));
    }

    #[test]
    fn chat_message_reasoning_empty_string_round_trips_but_omits_on_none() {
        // Some("") serializes as empty string (not omitted by is_none; caller should use None for empty).
        // The TUI fold normalizes empty to None before storing.
        let msg_empty = ChatMessage {
            role: "assistant".into(),
            content: "answer".into(),
            reasoning_content: Some(String::new()),
        };
        let v = serde_json::to_value(&msg_empty).unwrap();
        assert_eq!(v["reasoning_content"], "");
    }
}
