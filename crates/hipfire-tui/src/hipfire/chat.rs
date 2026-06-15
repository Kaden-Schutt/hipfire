// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    io::{BufRead, BufReader},
    sync::mpsc::Sender,
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
    Status(String),
    Done,
    Error(String),
}

pub fn stream_chat(
    host: &str,
    port: u16,
    model: &str,
    messages: &[ChatMessage],
    tx: Sender<ChatEvent>,
) -> Result<()> {
    let result = stream_chat_inner(host, port, model, messages, &tx);
    if let Err(err) = result {
        let _ = tx.send(ChatEvent::Error(err.to_string()));
    }
    Ok(())
}

fn stream_chat_inner(
    host: &str,
    port: u16,
    model: &str,
    messages: &[ChatMessage],
    tx: &Sender<ChatEvent>,
) -> Result<()> {
    let url = format!("http://{host}:{port}/v1/chat/completions");
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(600))
        .build();
    let body = json!({
        "model": model,
        "stream": true,
        "messages": messages,
    });
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
    let _ = tx.send(ChatEvent::Status("connected; waiting for tokens".into()));
    for line in reader.lines() {
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
        if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str) {
            let _ = tx.send(ChatEvent::Delta(text.to_string()));
        }
        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            let _ = tx.send(ChatEvent::Delta(text.to_string()));
        }
    }

    let _ = tx.send(ChatEvent::Done);
    Ok(())
}
