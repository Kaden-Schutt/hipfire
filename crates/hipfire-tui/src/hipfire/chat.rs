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
}

#[derive(Debug)]
pub enum ChatEvent {
    Delta(String),
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
                OpenAiSseEvent::Reasoning { text } | OpenAiSseEvent::Content { text } => {
                    let _ = tx.send(ChatEvent::Delta(text));
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
