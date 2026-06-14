// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Daemon JSONL protocol contracts.

use serde::{Deserialize, Serialize};

pub use hipfire_generate::{
    DoneEvent as DoneResponse, ErrorEvent as ErrorResponse, GenerateTextRequest as GenerateRequest,
    GenerationSamplingPolicy, TokenEvent as TokenResponse,
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonRequest {
    Load(LoadRequest),
    Unload,
    Ping,
    Generate(GenerateRequest),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LoadRequest {
    pub model: String,
    pub params: LoadParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
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
            sampling: GenerationSamplingPolicy {
                temperature: 0.7,
                max_tokens: 32,
                top_p: None,
                repeat_penalty: None,
            },
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
    fn generate_request_deserializes_jsonl_wire_shape() {
        let req: DaemonRequest = serde_json::from_value(json!({
            "type": "generate",
            "id": "req-1",
            "prompt": "last user text",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "last user text"}
            ],
            "temperature": 0.7,
            "max_tokens": 32,
            "top_p": 0.8,
            "worker_key_id": "worker-a",
            "ignored_legacy_field": true
        }))
        .unwrap();

        let DaemonRequest::Generate(req) = req else {
            panic!("expected generate request");
        };
        assert_eq!(req.id, "req-1");
        assert_eq!(req.prompt, "last user text");
        assert_eq!(req.messages.as_ref().unwrap()[0].role, Role::System);
        assert_eq!(req.sampling.top_p, Some(0.8));
        assert_eq!(req.worker_key_id.as_deref(), Some("worker-a"));
    }

    #[test]
    fn load_request_deserializes_jsonl_wire_shape() {
        let req: DaemonRequest = serde_json::from_value(json!({
            "type": "load",
            "model": "model.hfq",
            "params": {
                "max_seq": 4096,
                "physical_cap": 2048,
                "kv_cache": "fp16",
                "dflash_mode": "off",
                "draft": "draft.hfq",
                "cask_sidecar": "sidecar.triattn.hfq",
                "ignored": true
            },
            "request_id": "load-1",
            "ignored_legacy_field": true
        }))
        .unwrap();

        let DaemonRequest::Load(req) = req else {
            panic!("expected load request");
        };
        assert_eq!(req.model, "model.hfq");
        assert_eq!(req.params.max_seq, 4096);
        assert_eq!(req.params.physical_cap, Some(2048));
        assert_eq!(req.params.kv_cache.as_deref(), Some("fp16"));
        assert_eq!(req.params.dflash_mode.as_deref(), Some("off"));
        assert_eq!(req.params.draft.as_deref(), Some("draft.hfq"));
        assert_eq!(
            req.params.cask_sidecar.as_deref(),
            Some("sidecar.triattn.hfq")
        );
        assert_eq!(req.request_id.as_deref(), Some("load-1"));
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
