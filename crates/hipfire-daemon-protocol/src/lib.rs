// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Daemon JSONL protocol contracts.

use serde::{Deserialize, Serialize};

use hipfire_generate::{DoneEvent, ErrorEvent, GenerateTextRequest, TokenEvent, ToolCallsEvent};
use hipfire_model::{AcceleratorInventory, ModelLoadRequest, ModelLoadedResponse};

#[derive(Debug, Deserialize, Serialize)]
pub struct RequestControl {
    pub id: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonRequest {
    Load(ModelLoadRequest),
    Unload,
    Reset,
    Ping,
    Inventory,
    Generate(GenerateTextRequest),
    Abort(RequestControl),
    ForceAnswer(RequestControl),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonResponse {
    Loaded(ModelLoadedResponse),
    Unloaded,
    Reset,
    Pong,
    Inventory(AcceleratorInventory),
    Token(TokenEvent),
    ToolCalls(ToolCallsEvent),
    Done(DoneEvent),
    Error(ErrorEvent),
    #[serde(other)]
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_generate::{GenerateTextRequest, GenerationSamplingPolicy};
    use hipfire_prompt::{Message, Role};
    use serde_json::json;

    #[test]
    fn generate_request_serializes_structured_messages_without_nested_prompt() {
        let req = DaemonRequest::Generate(GenerateTextRequest {
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
            stop: None,
            image_base64: None,
            thinking: None,
            thinking_mode: None,
            reasoning_effort: None,
            assistant_prefix: None,
            max_think_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            request_id: None,
            evidence_dir: None,
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
    fn request_control_messages_match_bun_wire_shape() {
        let abort = serde_json::to_value(DaemonRequest::Abort(RequestControl {
            id: "chatcmpl-1".to_string(),
        }))
        .unwrap();
        assert_eq!(abort, json!({"type": "abort", "id": "chatcmpl-1"}));

        let force = serde_json::to_value(DaemonRequest::ForceAnswer(RequestControl {
            id: "chatcmpl-1".to_string(),
        }))
        .unwrap();
        assert_eq!(force, json!({"type": "force_answer", "id": "chatcmpl-1"}));

        let parsed: DaemonRequest = serde_json::from_value(json!({
            "type": "abort",
            "id": "chatcmpl-1",
            "ignored": true
        }))
        .unwrap();
        let DaemonRequest::Abort(control) = parsed else {
            panic!("expected abort request");
        };
        assert_eq!(control.id, "chatcmpl-1");
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
    fn reset_request_and_response_preserve_existing_wire_shape() {
        let value = serde_json::to_value(DaemonRequest::Reset).unwrap();
        assert_eq!(value, json!({"type": "reset"}));

        let response: DaemonResponse =
            serde_json::from_value(json!({"type": "reset", "seq_pos": 0})).unwrap();
        assert!(matches!(response, DaemonResponse::Reset));
    }

    #[test]
    fn inventory_request_and_response_use_shared_model_contract() {
        let value = serde_json::to_value(DaemonRequest::Inventory).unwrap();
        assert_eq!(value, json!({"type": "inventory"}));

        let response: DaemonResponse = serde_json::from_value(json!({
            "type": "inventory",
            "source": "daemon",
            "devices": [{
                "kind": "hip",
                "device_id": "0",
                "ordinal": 0,
                "arch": "gfx1201",
                "name": null,
                "total_memory_bytes": 24000000000u64,
                "integrated": false,
                "runtime": "HIP 6.4",
                "available": true,
                "selected": true,
                "reason": null
            }, {
                "kind": "npu",
                "device_id": "xdna1:0",
                "ordinal": 0,
                "arch": "xdna1",
                "name": "XDNA1 NPU",
                "total_memory_bytes": null,
                "integrated": null,
                "runtime": "xdna1_ffi",
                "available": true,
                "selected": false,
                "reason": null
            }]
        }))
        .unwrap();

        let DaemonResponse::Inventory(inventory) = response else {
            panic!("expected inventory response");
        };
        assert_eq!(inventory.source, "daemon");
        assert_eq!(inventory.devices.len(), 2);
        assert_eq!(inventory.devices[0].device_id, "0");
        assert_eq!(inventory.devices[0].arch.as_deref(), Some("gfx1201"));
        assert_eq!(inventory.devices[1].kind, "npu");
        assert_eq!(inventory.devices[1].device_id, "xdna1:0");
        assert_eq!(inventory.devices[1].arch.as_deref(), Some("xdna1"));
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
                "mtp_mode": "auto",
                "mtp_k": 3,
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
        assert_eq!(req.params.mtp_mode.as_deref(), Some("auto"));
        assert_eq!(req.params.mtp_k, Some(3));
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

    #[test]
    fn tool_calls_response_deserializes_daemon_shape() {
        let response: DaemonResponse = serde_json::from_value(json!({
            "type": "tool_calls",
            "id": "req-1",
            "calls": [{"name": "lookup", "arguments": {"q": "hipfire"}}]
        }))
        .unwrap();

        let DaemonResponse::ToolCalls(tool_calls) = response else {
            panic!("expected tool_calls response");
        };
        assert_eq!(tool_calls.id, "req-1");
        assert_eq!(tool_calls.calls[0].name, "lookup");
        assert_eq!(tool_calls.calls[0].arguments, json!({"q": "hipfire"}));
    }
}
