// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Typed generation request, event, and batch-plan contracts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GenerationSamplingPolicy {
    pub temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    pub max_tokens: u32,
}

impl GenerationSamplingPolicy {
    pub fn greedy(max_tokens: u32) -> Self {
        Self {
            temperature: 0.0,
            top_p: None,
            repeat_penalty: None,
            max_tokens,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateTextRequest {
    pub id: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<hipfire_prompt::Message>>,
    #[serde(flatten)]
    pub sampling: GenerationSamplingPolicy,
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GenerationEvent {
    Token(TokenEvent),
    Done(DoneEvent),
    Error(ErrorEvent),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TokenEvent {
    pub id: String,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DoneEvent {
    pub id: String,
    pub tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tok_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_tok_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_tok_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ErrorEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateBatchPrefillEnvelope {
    pub id: String,
    pub batch_id: String,
    pub session_count: usize,
    pub sessions: Vec<GenerateBatchPrefillSession>,
}

impl GenerateBatchPrefillEnvelope {
    pub fn is_probe(&self) -> bool {
        self.id == "prefill-batch-probe" && self.batch_id == "prefill-batch-probe"
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateBatchPrefillSession {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix_tokens: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages_history: Option<Vec<hipfire_prompt::Message>>,
    pub assistant_prefix: String,
    pub max_think_tokens: usize,
    pub semantic_boundary_checkpoints: bool,
    pub state_handle: GenerateBatchPrefillStateHandle,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct GenerateBatchPrefillStateHandle {
    pub state_kinds: Vec<String>,
    pub logical_position: usize,
    pub cached_prefix_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_state_handle: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_hash: Option<GenerateBatchPrefillPrefixHash>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct GenerateBatchPrefillPrefixHash {
    pub algorithm: String,
    pub value: String,
    pub prefix_len: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixHashPreflightEnvelope {
    pub id: String,
    pub boundary_policy: String,
    pub session: GenerateBatchPrefillSession,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct PrefixHashPreflightCandidate {
    pub hash: GenerateBatchPrefillPrefixHash,
    pub boundary: String,
    pub boundary_index: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct GenerateBatchDecodeEnvelope {
    pub id: String,
    pub batch_id: String,
    pub session_count: usize,
    pub cached_prefix_tokens: usize,
    pub sessions: Vec<GenerateBatchDecodeSession>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct GenerateBatchDecodeSession {
    pub id: String,
    pub session_id: String,
    pub max_tokens_remaining: usize,
    pub logical_position: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerateBatchPrefillPlan {
    SerialExact,
    FusedDenseQwen35Candidate,
    GroupedMoeQwen35Candidate,
}

impl GenerateBatchPrefillPlan {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SerialExact => "serial_exact",
            Self::FusedDenseQwen35Candidate => "fused_dense_qwen35_candidate",
            Self::GroupedMoeQwen35Candidate => "grouped_moe_qwen35_candidate",
        }
    }
}

pub fn plan_generate_batch_prefill_qwen35(
    arch_id: u32,
    session_count: usize,
) -> GenerateBatchPrefillPlan {
    match (arch_id, session_count > 1) {
        (5, true) => GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
        (6, true) => GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
        _ => GenerateBatchPrefillPlan::SerialExact,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35PrefillBatchBackend {
    SerialReference,
    FusedDense,
    FusedGroupedMoe,
}

impl Qwen35PrefillBatchBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SerialReference => "serial_reference",
            Self::FusedDense => "fused_dense",
            Self::FusedGroupedMoe => "fused_grouped_moe",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35DecodeBatchBackend {
    SerialReference,
    FusedDenseLayerChunked,
    FusedGroupedMoeLayerChunked,
}

impl Qwen35DecodeBatchBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SerialReference => "serial_reference",
            Self::FusedDenseLayerChunked => "fused_dense_layer_chunked",
            Self::FusedGroupedMoeLayerChunked => "fused_grouped_moe_layer_chunked",
        }
    }
}

pub fn select_qwen35_decode_batch_backend(
    requested: &str,
    arch_id: u32,
    session_count: usize,
) -> Result<Qwen35DecodeBatchBackend, String> {
    match requested {
        "" | "auto" | "serial" | "serial_reference" => {
            Ok(Qwen35DecodeBatchBackend::SerialReference)
        }
        "off" => Err(
            "generate_batch_decode_step disabled by HIPFIRE_QWEN35_DECODE_BATCH=off".to_string(),
        ),
        "fused" | "fused_dense" | "fused_dense_layer_chunked" => {
            if arch_id != 5 {
                return Err(format!(
                    "qwen35 fused dense decode batch requested, but arch_id={arch_id} is not dense Qwen35"
                ));
            }
            Ok(Qwen35DecodeBatchBackend::FusedDenseLayerChunked)
        }
        "fused_grouped_moe" | "grouped_moe" | "fused_grouped_moe_layer_chunked" => {
            if arch_id != 6 {
                return Err(format!(
                    "qwen35 grouped-MoE decode batch requested, but arch_id={arch_id} is not Qwen35 grouped-MoE"
                ));
            }
            if session_count < 2 {
                return Err(
                    "qwen35 grouped-MoE decode batch requires at least two sessions".to_string(),
                );
            }
            Ok(Qwen35DecodeBatchBackend::FusedGroupedMoeLayerChunked)
        }
        other => Err(format!(
            "unsupported HIPFIRE_QWEN35_DECODE_BATCH={other}; expected auto, serial, fused, fused_grouped_moe, or off"
        )),
    }
}

pub fn qwen35_decode_batch_requested_auto(requested: &str) -> bool {
    matches!(requested, "" | "auto")
}

pub fn qwen35_grouped_moe_decode_auto_latency_gate_passed(session_count: usize) -> bool {
    session_count >= 4
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35DecodeBatchSchedulerMetadata {
    pub selected_backend: &'static str,
    pub batch_size: usize,
    pub compatible_state_kinds: Vec<&'static str>,
    pub cached_prefix_tokens: usize,
    pub fallback_reason: &'static str,
}

pub fn qwen35_decode_batch_scheduler_metadata(
    requested: &str,
    arch_id: u32,
    backend: Qwen35DecodeBatchBackend,
    batch_size: usize,
    cached_prefix_tokens: usize,
) -> Qwen35DecodeBatchSchedulerMetadata {
    let fallback_reason = if qwen35_decode_batch_requested_auto(requested) {
        match (arch_id, backend) {
            (6, Qwen35DecodeBatchBackend::SerialReference)
                if !qwen35_grouped_moe_decode_auto_latency_gate_passed(batch_size) =>
            {
                "auto_grouped_moe_serial_small_batch_latency_gate"
            }
            (6, Qwen35DecodeBatchBackend::SerialReference) => {
                "auto_grouped_moe_serial_pending_latency_gate"
            }
            (_, Qwen35DecodeBatchBackend::SerialReference) if batch_size < 2 => {
                "auto_requires_multi_session"
            }
            (_, Qwen35DecodeBatchBackend::SerialReference) => "auto_serial_reference",
            _ => "none",
        }
    } else if backend == Qwen35DecodeBatchBackend::SerialReference {
        "requested_serial_reference"
    } else {
        "none"
    };
    Qwen35DecodeBatchSchedulerMetadata {
        selected_backend: backend.as_str(),
        batch_size,
        compatible_state_kinds: vec!["attention_kv", "deltanet_recurrent"],
        cached_prefix_tokens,
        fallback_reason,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationTiming {
    pub prefill_tokens: Option<u32>,
    pub prefill_ms: Option<u64>,
    pub prefill_tok_s: Option<u64>,
    pub decode_tok_s: Option<u64>,
    pub ttft_ms: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefill_session(id: &str) -> GenerateBatchPrefillSession {
        GenerateBatchPrefillSession {
            id: id.to_string(),
            prompt: Some("hello".to_string()),
            suffix_tokens: None,
            system_prompt: None,
            tools: None,
            messages_history: None,
            assistant_prefix: "<|im_start|>assistant\n".to_string(),
            max_think_tokens: 0,
            semantic_boundary_checkpoints: false,
            state_handle: GenerateBatchPrefillStateHandle {
                state_kinds: vec!["attention_kv".to_string(), "deltanet_recurrent".to_string()],
                logical_position: 0,
                cached_prefix_tokens: 0,
                runtime_state_handle: None,
                prefix_hash: None,
            },
        }
    }

    #[test]
    fn generation_events_serialize_to_daemon_jsonl_shapes() {
        let token = GenerationEvent::Token(TokenEvent {
            id: "r1".to_string(),
            text: "The".to_string(),
        });
        assert_eq!(
            serde_json::to_value(token).unwrap(),
            serde_json::json!({"type": "token", "id": "r1", "text": "The"})
        );

        let done = GenerationEvent::Done(DoneEvent {
            id: "r1".to_string(),
            tokens: 42,
            tok_s: Some(44.5),
            prefill_tokens: None,
            prefill_ms: None,
            prefill_tok_s: None,
            decode_tok_s: None,
            ttft_ms: None,
            finish_reason: Some("stop".to_string()),
            response_id: None,
            extra: HashMap::new(),
        });
        assert_eq!(
            serde_json::to_value(done).unwrap(),
            serde_json::json!({
                "type": "done",
                "id": "r1",
                "tokens": 42,
                "tok_s": 44.5,
                "finish_reason": "stop"
            })
        );
    }

    #[test]
    fn generate_text_request_preserves_server_daemon_contract() {
        let req = GenerateTextRequest {
            id: "chatcmpl-1".to_string(),
            prompt: "Hello".to_string(),
            messages: None,
            sampling: GenerationSamplingPolicy {
                temperature: 0.3,
                top_p: Some(0.8),
                repeat_penalty: Some(1.05),
                max_tokens: 128,
            },
            worker_key_id: Some("worker-a".to_string()),
            tools: None,
            system: None,
            thinking: Some("auto".to_string()),
            max_think_tokens: Some(64),
            request_id: Some("req-1".to_string()),
        };
        assert_eq!(
            serde_json::to_value(req).unwrap(),
            serde_json::json!({
                "id": "chatcmpl-1",
                "prompt": "Hello",
                "temperature": 0.3,
                "top_p": 0.8,
                "repeat_penalty": 1.05,
                "max_tokens": 128,
                "worker_key_id": "worker-a",
                "thinking": "auto",
                "max_think_tokens": 64,
                "request_id": "req-1"
            })
        );
    }

    #[test]
    fn prefill_envelope_probe_and_json_shape_are_stable() {
        let envelope = GenerateBatchPrefillEnvelope {
            id: "prefill-batch-probe".to_string(),
            batch_id: "prefill-batch-probe".to_string(),
            session_count: 1,
            sessions: vec![prefill_session("s1")],
        };
        assert!(envelope.is_probe());
        let value = serde_json::to_value(&envelope).unwrap();
        assert_eq!(value["sessions"][0]["state_handle"]["logical_position"], 0);
        assert_eq!(
            value["sessions"][0]["assistant_prefix"],
            "<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen35_prefill_plans_match_daemon_arch_rules() {
        assert_eq!(
            plan_generate_batch_prefill_qwen35(5, 2),
            GenerateBatchPrefillPlan::FusedDenseQwen35Candidate
        );
        assert_eq!(
            plan_generate_batch_prefill_qwen35(6, 2),
            GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate
        );
        assert_eq!(
            plan_generate_batch_prefill_qwen35(6, 1),
            GenerateBatchPrefillPlan::SerialExact
        );
        assert_eq!(
            GenerateBatchPrefillPlan::SerialExact.as_str(),
            "serial_exact"
        );
    }

    #[test]
    fn qwen35_decode_backend_selection_matches_daemon_policy() {
        assert_eq!(
            select_qwen35_decode_batch_backend("auto", 6, 4).unwrap(),
            Qwen35DecodeBatchBackend::SerialReference
        );
        assert_eq!(
            select_qwen35_decode_batch_backend("fused", 5, 1).unwrap(),
            Qwen35DecodeBatchBackend::FusedDenseLayerChunked
        );
        assert_eq!(
            select_qwen35_decode_batch_backend("grouped_moe", 6, 2).unwrap(),
            Qwen35DecodeBatchBackend::FusedGroupedMoeLayerChunked
        );
        assert!(select_qwen35_decode_batch_backend("grouped_moe", 6, 1)
            .unwrap_err()
            .contains("at least two sessions"));
    }

    #[test]
    fn qwen35_decode_scheduler_metadata_reports_fallback_reason() {
        let metadata = qwen35_decode_batch_scheduler_metadata(
            "auto",
            6,
            Qwen35DecodeBatchBackend::SerialReference,
            2,
            16,
        );
        assert_eq!(
            metadata.fallback_reason,
            "auto_grouped_moe_serial_small_batch_latency_gate"
        );
        assert_eq!(
            metadata.compatible_state_kinds,
            vec!["attention_kv", "deltanet_recurrent"]
        );

        let requested = qwen35_decode_batch_scheduler_metadata(
            "serial",
            5,
            Qwen35DecodeBatchBackend::SerialReference,
            2,
            8,
        );
        assert_eq!(requested.fallback_reason, "requested_serial_reference");
    }
}
