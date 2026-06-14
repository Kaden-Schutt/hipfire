// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Daemon JSONL protocol contracts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonRequest {
    Load(LoadRequest),
    Unload,
    Ping,
    Generate(GenerateRequest),
}

#[derive(Debug, Serialize)]
pub struct LoadRequest {
    pub model: String,
    pub params: LoadParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct LoadParams {
    pub max_seq: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_cap: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dflash_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cask_sidecar: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GenerateRequest {
    pub id: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<hipfire_prompt::Message>>,
    pub temperature: f64,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_key_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_think_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonResponse {
    Loaded(LoadedResponse),
    Unloaded,
    Pong,
    Token(TokenResponse),
    Done(DoneResponse),
    Error(ErrorResponse),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct LoadedResponse {
    pub worker_key_id: String,
    pub arch: Option<String>,
    pub dim: Option<u32>,
    pub layers: Option<u32>,
    pub vocab: Option<u32>,
    pub model_worker: Option<serde_json::Value>,
    #[serde(default)]
    pub response_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TokenResponse {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct DoneResponse {
    pub id: String,
    pub tokens: u32,
    pub tok_s: Option<f64>,
    pub prefill_tokens: Option<u32>,
    pub prefill_ms: Option<f64>,
    pub prefill_tok_s: Option<f64>,
    pub decode_tok_s: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub response_id: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ErrorResponse {
    pub id: Option<String>,
    pub message: String,
    #[serde(default)]
    pub response_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_prompt::{Message, Role};
    use serde_json::json;

    #[test]
    fn generate_request_serializes_structured_messages_without_nested_prompt() {
        let req = DaemonRequest::Generate(GenerateRequest {
            id: "req-1".to_string(),
            prompt: "last user text".to_string(),
            messages: Some(vec![
                Message {
                    role: Role::System,
                    content: "be brief".to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                },
                Message {
                    role: Role::User,
                    content: "last user text".to_string(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                },
            ]),
            temperature: 0.7,
            max_tokens: 32,
            top_p: None,
            repeat_penalty: None,
            worker_key_id: Some("worker-a".to_string()),
            tools: None,
            system: None,
            thinking: None,
            max_think_tokens: None,
            request_id: None,
        });

        let value = serde_json::to_value(req).unwrap();
        assert_eq!(value["type"], "generate");
        assert_eq!(value["prompt"], "last user text");
        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(value["messages"][1]["content"], "last user text");
        assert!(value.get("tools").is_none());
        assert!(value.get("request_id").is_none());
    }

    #[test]
    fn done_response_preserves_unknown_metrics_in_extra() {
        let done: DaemonResponse = serde_json::from_value(json!({
            "type": "done",
            "id": "req-1",
            "tokens": 5,
            "tok_s": 10.5,
            "finish_reason": "stop",
            "backend_path": "hip_rdna_compute"
        }))
        .unwrap();

        let DaemonResponse::Done(done) = done else {
            panic!("expected done response");
        };
        assert_eq!(done.id, "req-1");
        assert_eq!(done.extra["backend_path"], "hip_rdna_compute");
    }
}
