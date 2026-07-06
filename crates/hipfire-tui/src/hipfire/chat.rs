// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    io::{BufRead, BufReader},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
        Arc,
    },
    time::Duration,
};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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
    let url = format!("http://{host}:{port}/v1/chat/completions");
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(600))
        // Per-read bound: a stalled socket (half-open TCP, server hang) errors out
        // within this window instead of parking the worker thread for the full
        // total timeout — paired with the optimistic UI abort in request_abort.
        .timeout_read(Duration::from_secs(120))
        .build();
    let mut body = json!({
        "model": model,
        "stream": true,
        "messages": messages,
    });
    // Per-session sampling overrides (set via /temp and /top_p).
    if let Some(t) = temperature {
        body["temperature"] = json!(t);
    }
    if let Some(p) = top_p {
        body["top_p"] = json!(p);
    }
    let resp = match agent
        .post(&url)
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
    {
        Ok(resp) => resp,
        Err(ureq::Error::Status(code, resp)) => {
            let text = resp.into_string().unwrap_or_default();
            return Err(anyhow!(
                "HTTP {code}: {}",
                text.chars().take(240).collect::<String>()
            ));
        }
        Err(err) => return Err(anyhow!(err.to_string())),
    };

    let reader = BufReader::new(resp.into_reader());
    for line in reader.lines() {
        // Cooperative cancel: checked once per streamed line, so an in-flight
        // generation stops within ~one token of the user pressing Esc. The
        // partial reply already streamed stays on screen.
        if abort.load(Ordering::Relaxed) {
            let _ = tx.send(ChatEvent::Done);
            return Ok(());
        }
        let line = line?;
        let trimmed = line.trim();
        if !trimmed.starts_with("data:") {
            continue;
        }
        let payload = trimmed.trim_start_matches("data:").trim();
        if payload == "[DONE]" {
            let _ = tx.send(ChatEvent::Done);
            return Ok(());
        }
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        if let Some(err) = value.get("error") {
            let msg = err
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| err.as_str())
                .unwrap_or("server error")
                .to_string();
            return Err(anyhow!(msg));
        }
        let Some(delta) = value
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"))
        else {
            continue;
        };
        // Coalesce reasoning + content into ONE Delta per chunk so the UI's
        // delta-count token proxy isn't double-counted for reasoning models.
        let mut chunk = String::new();
        if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str) {
            chunk.push_str(text);
        }
        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            chunk.push_str(text);
        }
        if !chunk.is_empty() {
            let _ = tx.send(ChatEvent::Delta(chunk));
        }
    }

    let _ = tx.send(ChatEvent::Done);
    Ok(())
}
