// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Typed generation request, event, and batch-plan contracts.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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

pub fn generate_prefix_hash_json(hash: &GenerateBatchPrefillPrefixHash) -> serde_json::Value {
    serde_json::json!({
        "algorithm": hash.algorithm,
        "value": hash.value,
        "prefix_len": hash.prefix_len,
    })
}

fn push_hash_field(buf: &mut Vec<u8>, label: &str, value: &str) {
    buf.extend_from_slice(label.as_bytes());
    buf.push(b'=');
    buf.extend_from_slice(value.as_bytes());
    buf.push(0);
}

pub fn compute_qwen35_prefix_hash(
    arch_id: u32,
    kv_mode: Option<&str>,
    state_kinds: &[String],
    assistant_prefix: &str,
    max_think_tokens: usize,
    tokens: &[u32],
) -> GenerateBatchPrefillPrefixHash {
    let mut buf = Vec::with_capacity(128 + tokens.len() * 4);
    push_hash_field(
        &mut buf,
        "domain",
        "hipfire.generate_batch_prefill.prefix.v1",
    );
    push_hash_field(&mut buf, "algorithm", "xxh128");
    push_hash_field(&mut buf, "arch_id", &arch_id.to_string());
    push_hash_field(&mut buf, "kv_mode", kv_mode.unwrap_or("unknown"));
    push_hash_field(&mut buf, "assistant_prefix", assistant_prefix);
    push_hash_field(&mut buf, "max_think_tokens", &max_think_tokens.to_string());
    let mut normalized_kinds = state_kinds.to_vec();
    normalized_kinds.sort();
    normalized_kinds.dedup();
    push_hash_field(&mut buf, "state_kinds", &normalized_kinds.join("+"));
    push_hash_field(&mut buf, "token_encoding", "u32le");
    for token in tokens {
        buf.extend_from_slice(&token.to_le_bytes());
    }
    let value = twox_hash::XxHash3_128::oneshot(&buf);
    GenerateBatchPrefillPrefixHash {
        algorithm: "xxh128".to_string(),
        value: format!("{value:032x}"),
        prefix_len: tokens.len(),
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct Qwen35SemanticBoundaryCheckpoint {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_id: Option<String>,
    pub prefix_len: usize,
    pub hash: GenerateBatchPrefillPrefixHash,
    pub boundary: String,
    pub boundary_index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35PrefillCheckpointKind<'a> {
    Final,
    SemanticBoundary {
        boundary: &'a str,
        boundary_index: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35PrefillCheckpointHook<'a> {
    pub batch_id: &'a str,
    pub session_id: &'a str,
    pub source_state_handle: &'a str,
    pub logical_position: usize,
    pub kind: Qwen35PrefillCheckpointKind<'a>,
    pub prefix_hash: &'a GenerateBatchPrefillPrefixHash,
}

pub fn qwen35_checkpoint_session_id(
    batch_id: &str,
    session_id: &str,
    logical_position: usize,
) -> String {
    format!("qwen35-checkpoint:{batch_id}:{session_id}:{logical_position}")
}

pub fn qwen35_boundary_checkpoint_session_id(
    batch_id: &str,
    session_id: &str,
    logical_position: usize,
    boundary_index: usize,
) -> String {
    format!(
        "qwen35-checkpoint:{batch_id}:{session_id}:boundary:{boundary_index}:{logical_position}"
    )
}

pub fn qwen35_prefill_checkpoint_session_id(hook: Qwen35PrefillCheckpointHook<'_>) -> String {
    match hook.kind {
        Qwen35PrefillCheckpointKind::Final => {
            qwen35_checkpoint_session_id(hook.batch_id, hook.session_id, hook.logical_position)
        }
        Qwen35PrefillCheckpointKind::SemanticBoundary { boundary_index, .. } => {
            qwen35_boundary_checkpoint_session_id(
                hook.batch_id,
                hook.session_id,
                hook.logical_position,
                boundary_index,
            )
        }
    }
}

pub fn qwen35_prefill_checkpoint_boundary_kind(hook: Qwen35PrefillCheckpointHook<'_>) -> &'_ str {
    match hook.kind {
        Qwen35PrefillCheckpointKind::Final => "full",
        Qwen35PrefillCheckpointKind::SemanticBoundary { boundary, .. } => boundary,
    }
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

fn parse_u32_array(value: &serde_json::Value, field: &str) -> Result<Vec<u32>, String> {
    let arr = value
        .as_array()
        .ok_or_else(|| format!("{field} must be an array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        match item.as_u64() {
            Some(n) if n <= u32::MAX as u64 => out.push(n as u32),
            _ => return Err(format!("{field}[{i}] must be a u32 token id")),
        }
    }
    Ok(out)
}

fn parse_prefix_hash(
    value: &serde_json::Value,
    field: &str,
) -> Result<GenerateBatchPrefillPrefixHash, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| format!("{field} must be an object"))?;
    let algorithm = obj
        .get("algorithm")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{field}.algorithm must be a string"))?;
    if algorithm != "xxh128" {
        return Err(format!("{field}.algorithm must be 'xxh128'"));
    }
    let hash_value = obj
        .get("value")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{field}.value must be a string"))?;
    if hash_value.len() != 32
        || !hash_value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(format!("{field}.value must be 32 lowercase hex characters"));
    }
    let prefix_len =
        obj.get("prefix_len")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("{field}.prefix_len must be an integer >= 0"))? as usize;
    Ok(GenerateBatchPrefillPrefixHash {
        algorithm: algorithm.to_string(),
        value: hash_value.to_string(),
        prefix_len,
    })
}

pub fn validate_generate_batch_prefill(
    msg: &serde_json::Value,
) -> Result<GenerateBatchPrefillEnvelope, String> {
    let id = msg
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("0")
        .to_string();
    let batch_id = msg
        .get("batch_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "generate_batch_prefill.batch_id must be a non-empty string".to_string())?
        .to_string();

    let has_worker = msg
        .get("worker_key_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .is_some()
        || msg
            .get("model")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .is_some();
    if !has_worker {
        return Err("generate_batch_prefill requires worker_key_id or model identity".to_string());
    }

    let sessions = msg
        .get("sessions")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "generate_batch_prefill.sessions must be an array".to_string())?;
    if sessions.is_empty() {
        return Err("generate_batch_prefill.sessions must not be empty".to_string());
    }

    let mut seen_session_ids = HashSet::new();
    let mut parsed_sessions = Vec::with_capacity(sessions.len());
    for (i, session) in sessions.iter().enumerate() {
        let prefix = format!("generate_batch_prefill.sessions[{i}]");
        session
            .as_object()
            .ok_or_else(|| format!("{prefix} must be an object"))?;
        let session_id = session
            .get("id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("{prefix}.id must be a non-empty string"))?;
        if !seen_session_ids.insert(session_id.to_string()) {
            return Err(format!(
                "generate_batch_prefill duplicate session id {session_id}"
            ));
        }

        let has_prompt = match session.get("prompt") {
            Some(v) => {
                v.as_str()
                    .ok_or_else(|| format!("{prefix}.prompt must be a string"))?;
                true
            }
            None => false,
        };
        let has_suffix = match session.get("suffix_tokens") {
            Some(v) => {
                parse_u32_array(v, &format!("{prefix}.suffix_tokens"))?;
                true
            }
            None => false,
        };
        if has_prompt == has_suffix {
            return Err(format!(
                "{prefix} must include exactly one of prompt or suffix_tokens"
            ));
        }

        let state_handle = session
            .get("state_handle")
            .and_then(|v| v.as_object())
            .ok_or_else(|| format!("{prefix}.state_handle must be an object"))?;
        let state_kinds = state_handle
            .get("state_kinds")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                format!("{prefix}.state_handle.state_kinds must be a non-empty array of strings")
            })?;
        if state_kinds.is_empty() {
            return Err(format!(
                "{prefix}.state_handle.state_kinds must be a non-empty array"
            ));
        }
        let valid_state_kinds = [
            "attention_kv",
            "deltanet_recurrent",
            "mamba_ssm",
            "mamba_conv",
            "architecture_specific",
        ];
        let mut parsed_state_kinds = Vec::with_capacity(state_kinds.len());
        for kind in state_kinds.iter() {
            let kind = kind
                .as_str()
                .ok_or_else(|| format!("{prefix}.state_handle.state_kinds must be strings"))?;
            if !valid_state_kinds.contains(&kind) {
                return Err(format!(
                    "{prefix}.state_handle.state_kinds contains unsupported kind {kind}"
                ));
            }
            parsed_state_kinds.push(kind.to_string());
        }
        let logical_position = state_handle
            .get("logical_position")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                format!("{prefix}.state_handle.logical_position must be an integer >= 0")
            })? as usize;
        let cached_prefix_tokens = if let Some(v) = state_handle.get("cached_prefix_tokens") {
            v.as_u64().ok_or_else(|| {
                format!("{prefix}.state_handle.cached_prefix_tokens must be an integer >= 0")
            })? as usize
        } else {
            0
        };
        let runtime_state_handle = match state_handle.get("runtime_state_handle") {
            Some(v) => Some(
                v.as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        format!(
                            "{prefix}.state_handle.runtime_state_handle must be a non-empty string"
                        )
                    })?
                    .to_string(),
            ),
            None => None,
        };
        let prefix_hash = match state_handle.get("prefix_hash") {
            Some(v) => Some(parse_prefix_hash(
                v,
                &format!("{prefix}.state_handle.prefix_hash"),
            )?),
            None => None,
        };
        if prefix_hash.is_some() && runtime_state_handle.is_none() {
            return Err(format!(
                "{prefix}.state_handle.prefix_hash requires runtime_state_handle"
            ));
        }
        if let Some(params) = session.get("params") {
            params
                .as_object()
                .ok_or_else(|| format!("{prefix}.params must be an object"))?;
        }
        let system_prompt = match session.get("system") {
            Some(v) => Some(
                v.as_str()
                    .ok_or_else(|| format!("{prefix}.system must be a string"))?
                    .to_string(),
            ),
            None => None,
        };
        let tools = match session.get("tools") {
            Some(v) => Some(
                serde_json::from_value::<Vec<serde_json::Value>>(v.clone())
                    .map_err(|e| format!("{prefix}.tools invalid: {e}"))?,
            ),
            None => None,
        };
        let messages_history = match session.get("messages") {
            Some(v) => Some(
                serde_json::from_value::<Vec<hipfire_prompt::Message>>(v.clone())
                    .map_err(|e| format!("{prefix}.messages invalid: {e}"))?,
            ),
            None => None,
        };
        let assistant_prefix = session
            .get("params")
            .and_then(|p| p.get("assistant_prefix"))
            .and_then(|v| v.as_str())
            .unwrap_or("plain")
            .to_string();
        let max_think_tokens = session
            .get("params")
            .and_then(|p| p.get("max_think_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let semantic_boundary_checkpoints = match session
            .get("params")
            .and_then(|p| p.get("semantic_boundary_checkpoints"))
        {
            Some(v) => v.as_bool().ok_or_else(|| {
                format!("{prefix}.params.semantic_boundary_checkpoints must be a boolean")
            })?,
            None => false,
        };

        parsed_sessions.push(GenerateBatchPrefillSession {
            id: session_id.to_string(),
            prompt: session
                .get("prompt")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            suffix_tokens: match session.get("suffix_tokens") {
                Some(v) => Some(parse_u32_array(v, &format!("{prefix}.suffix_tokens"))?),
                None => None,
            },
            system_prompt,
            tools,
            messages_history,
            assistant_prefix,
            max_think_tokens,
            semantic_boundary_checkpoints,
            state_handle: GenerateBatchPrefillStateHandle {
                state_kinds: parsed_state_kinds,
                logical_position,
                cached_prefix_tokens,
                runtime_state_handle,
                prefix_hash,
            },
        });
    }

    Ok(GenerateBatchPrefillEnvelope {
        id,
        batch_id,
        session_count: sessions.len(),
        sessions: parsed_sessions,
    })
}

pub fn validate_prefix_hash_preflight(
    msg: &serde_json::Value,
) -> Result<PrefixHashPreflightEnvelope, String> {
    let id = msg
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("prefix-hash-preflight")
        .to_string();
    let boundary_policy = msg
        .get("boundary_policy")
        .and_then(|v| v.as_str())
        .unwrap_or("semantic_chat_template")
        .to_string();
    if boundary_policy != "semantic_chat_template" {
        return Err(
            "prefix_hash_preflight.boundary_policy must be semantic_chat_template".to_string(),
        );
    }
    let session = msg
        .get("session")
        .ok_or_else(|| "prefix_hash_preflight.session must be an object".to_string())?;
    let worker_key = msg.get("worker_key_id").cloned();
    let model = msg.get("model").cloned();
    let mut generated = serde_json::json!({
        "type": "generate_batch_prefill",
        "id": id,
        "batch_id": id,
        "sessions": [session.clone()],
    });
    if let Some(worker_key) = worker_key {
        generated["worker_key_id"] = worker_key;
    }
    if let Some(model) = model {
        generated["model"] = model;
    }
    let envelope = validate_generate_batch_prefill(&generated)
        .map_err(|e| format!("prefix_hash_preflight.{e}"))?;
    let session = envelope
        .sessions
        .into_iter()
        .next()
        .ok_or_else(|| "prefix_hash_preflight.session missing after validation".to_string())?;
    if session.prompt.is_none() {
        return Err("prefix_hash_preflight.session must include prompt".to_string());
    }
    if session.state_handle.runtime_state_handle.is_some() {
        return Err(
            "prefix_hash_preflight.session must not include runtime_state_handle".to_string(),
        );
    }
    Ok(PrefixHashPreflightEnvelope {
        id,
        boundary_policy,
        session,
    })
}

pub fn validate_generate_batch_decode(
    msg: &serde_json::Value,
) -> Result<GenerateBatchDecodeEnvelope, String> {
    let id = msg
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("0")
        .to_string();
    let batch_id = msg
        .get("batch_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            "generate_batch_decode_step.batch_id must be a non-empty string".to_string()
        })?
        .to_string();
    let has_worker = msg
        .get("worker_key_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .is_some()
        || msg
            .get("model")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .is_some();
    if !has_worker {
        return Err(
            "generate_batch_decode_step requires worker_key_id or model identity".to_string(),
        );
    }
    let sessions = msg
        .get("sessions")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "generate_batch_decode_step.sessions must be an array".to_string())?;
    if sessions.is_empty() {
        return Err("generate_batch_decode_step.sessions must not be empty".to_string());
    }
    let cached_prefix_tokens = msg
        .get("cached_prefix_tokens")
        .map(|v| {
            v.as_u64().map(|n| n as usize).ok_or_else(|| {
                "generate_batch_decode_step.cached_prefix_tokens must be an integer >= 0"
                    .to_string()
            })
        })
        .transpose()?
        .unwrap_or(0);
    let mut seen_ids = HashSet::new();
    let mut parsed = Vec::with_capacity(sessions.len());
    for (i, session) in sessions.iter().enumerate() {
        let prefix = format!("generate_batch_decode_step.sessions[{i}]");
        session
            .as_object()
            .ok_or_else(|| format!("{prefix} must be an object"))?;
        let id = session
            .get("id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("{prefix}.id must be a non-empty string"))?;
        if !seen_ids.insert(id.to_string()) {
            return Err(format!(
                "generate_batch_decode_step duplicate session id {id}"
            ));
        }
        let session_id = session
            .get("session_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("{prefix}.session_id must be a non-empty string"))?;
        let max_tokens_remaining = session
            .get("max_tokens_remaining")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                format!("{prefix}.max_tokens_remaining must be an integer greater than 0")
            })? as usize;
        if max_tokens_remaining == 0 {
            return Err(format!(
                "{prefix}.max_tokens_remaining must be greater than 0"
            ));
        }
        let logical_position = session
            .get("logical_position")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("{prefix}.logical_position must be an integer >= 0"))?
            as usize;
        parsed.push(GenerateBatchDecodeSession {
            id: id.to_string(),
            session_id: session_id.to_string(),
            max_tokens_remaining,
            logical_position,
        });
    }
    Ok(GenerateBatchDecodeEnvelope {
        id,
        batch_id,
        cached_prefix_tokens,
        session_count: sessions.len(),
        sessions: parsed,
    })
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35PrefillSessionResult {
    pub id: String,
    pub prefill_tokens: usize,
    pub logical_position: usize,
    pub cached_prefix_tokens: usize,
    pub prefix_hash: GenerateBatchPrefillPrefixHash,
    pub debug_sample_token: Option<u32>,
    pub boundary_checkpoints: Vec<Qwen35SemanticBoundaryCheckpoint>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35PrefillBatchResult {
    pub mode: &'static str,
    pub plan: GenerateBatchPrefillPlan,
    pub backend: Qwen35PrefillBatchBackend,
    pub total_prefill_tokens: usize,
    pub sessions: Vec<Qwen35PrefillSessionResult>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35FusedDensePrefillInputKind {
    FullPrompt,
    GeneratedSuffixReplay,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35FusedDensePrefillSessionSpec<'a> {
    pub id: &'a str,
    pub tokens: &'a [u32],
    pub cached_prefix_tokens: usize,
    pub replay_as_generated_suffix: bool,
    pub state_kinds: &'a [String],
    pub assistant_prefix: &'a str,
    pub max_think_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35FusedDensePrefillBatchContract<'a> {
    pub input_kind: Qwen35FusedDensePrefillInputKind,
    pub total_tokens: usize,
    pub sessions: Vec<Qwen35FusedDensePrefillSessionSpec<'a>>,
}

pub fn select_qwen35_prefill_batch_backend(
    plan: GenerateBatchPrefillPlan,
    requested: Option<&str>,
    fused_grouped_moe_supported: Result<(), String>,
) -> Result<Qwen35PrefillBatchBackend, String> {
    match requested.unwrap_or("auto") {
        "auto" | "" => match plan {
            GenerateBatchPrefillPlan::FusedDenseQwen35Candidate => {
                Ok(Qwen35PrefillBatchBackend::FusedDense)
            }
            GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate => {
                if fused_grouped_moe_supported.is_ok() {
                    Ok(Qwen35PrefillBatchBackend::FusedGroupedMoe)
                } else {
                    Ok(Qwen35PrefillBatchBackend::SerialReference)
                }
            }
            GenerateBatchPrefillPlan::SerialExact => Ok(Qwen35PrefillBatchBackend::SerialReference),
        },
        "serial" | "serial_reference" => Ok(Qwen35PrefillBatchBackend::SerialReference),
        "fused" | "fused_dense" => {
            if plan == GenerateBatchPrefillPlan::FusedDenseQwen35Candidate {
                Ok(Qwen35PrefillBatchBackend::FusedDense)
            } else if plan == GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate {
                fused_grouped_moe_supported?;
                Ok(Qwen35PrefillBatchBackend::FusedGroupedMoe)
            } else {
                Err(format!(
                    "qwen35 fused prefill-session batch requested, but plan={} is not fused-eligible",
                    plan.as_str()
                ))
            }
        }
        "fused_moe" | "grouped_moe" | "fused_grouped_moe" => {
            if plan == GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate {
                fused_grouped_moe_supported?;
                Ok(Qwen35PrefillBatchBackend::FusedGroupedMoe)
            } else {
                Err(format!(
                    "qwen35 grouped-MoE fused prefill-session batch requested, but plan={} is not grouped-MoE eligible",
                    plan.as_str()
                ))
            }
        }
        other => Err(format!(
            "unsupported HIPFIRE_QWEN35_PREFILL_SESSION_BATCH={other}; expected auto, serial, fused, or fused_moe"
        )),
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

pub fn qwen35_prefill_scratch_target_batch(
    paged_experts: bool,
    required_rows: usize,
    configured_max_batch: Option<&str>,
    default_max_batch: usize,
) -> usize {
    let configured = configured_max_batch
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v >= 2);
    if let Some(configured) = configured {
        return configured.max(required_rows);
    }
    if paged_experts {
        return required_rows.max(2);
    }
    default_max_batch.max(required_rows)
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
    fn validates_generate_batch_prefill_json_contract() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "prefill-1",
            "batch_id": "batch-1",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "suffix_tokens": [4, 5],
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 3,
                    "cached_prefix_tokens": 3,
                    "runtime_state_handle": "qwen35-checkpoint:req-0",
                    "prefix_hash": {
                        "algorithm": "xxh128",
                        "value": "0123456789abcdef0123456789abcdef",
                        "prefix_len": 3
                    }
                },
                "params": {
                    "semantic_boundary_checkpoints": true
                }
            }]
        });

        let envelope = validate_generate_batch_prefill(&msg).expect("valid prefill envelope");
        assert_eq!(envelope.id, "prefill-1");
        assert_eq!(envelope.batch_id, "batch-1");
        assert_eq!(envelope.session_count, 1);
        assert!(envelope.sessions[0].semantic_boundary_checkpoints);
        assert_eq!(
            envelope.sessions[0]
                .state_handle
                .runtime_state_handle
                .as_deref(),
            Some("qwen35-checkpoint:req-0")
        );
        assert_eq!(
            envelope.sessions[0]
                .state_handle
                .prefix_hash
                .as_ref()
                .unwrap()
                .prefix_len,
            3
        );
    }

    #[test]
    fn validates_prefix_hash_preflight_json_contract() {
        let msg = serde_json::json!({
            "type": "prefix_hash_preflight",
            "id": "prefix-1",
            "worker_key_id": "worker-a",
            "boundary_policy": "semantic_chat_template",
            "session": {
                "id": "req-1",
                "prompt": "hello",
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hello"}
                ],
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 0
                },
                "params": {
                    "assistant_prefix": "open_think",
                    "max_think_tokens": 16
                }
            }
        });

        let envelope = validate_prefix_hash_preflight(&msg).expect("valid preflight envelope");
        assert_eq!(envelope.id, "prefix-1");
        assert_eq!(envelope.boundary_policy, "semantic_chat_template");
        assert_eq!(envelope.session.id, "req-1");
        assert_eq!(envelope.session.messages_history.as_ref().unwrap().len(), 2);
        assert_eq!(envelope.session.assistant_prefix, "open_think");
        assert_eq!(envelope.session.max_think_tokens, 16);
    }

    #[test]
    fn validates_generate_batch_decode_json_contract() {
        let msg = serde_json::json!({
            "type": "generate_batch_decode_step",
            "id": "decode-1",
            "batch_id": "decode-batch-1",
            "worker_key_id": "worker-a",
            "cached_prefix_tokens": 12,
            "sessions": [
                {
                    "id": "req-1",
                    "session_id": "qwen35-checkpoint:batch:req-1:8",
                    "logical_position": 8,
                    "max_tokens_remaining": 4
                },
                {
                    "id": "req-2",
                    "session_id": "qwen35-checkpoint:batch:req-2:8",
                    "logical_position": 8,
                    "max_tokens_remaining": 3
                }
            ]
        });

        let envelope = validate_generate_batch_decode(&msg).expect("valid decode envelope");
        assert_eq!(envelope.id, "decode-1");
        assert_eq!(envelope.batch_id, "decode-batch-1");
        assert_eq!(envelope.session_count, 2);
        assert_eq!(envelope.cached_prefix_tokens, 12);
        assert_eq!(
            envelope.sessions[0].session_id,
            "qwen35-checkpoint:batch:req-1:8"
        );
        assert_eq!(envelope.sessions[1].max_tokens_remaining, 3);
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
    fn qwen35_prefix_hash_is_domain_separated() {
        let kinds = vec!["deltanet_recurrent".to_string(), "attention_kv".to_string()];
        let reordered = vec!["attention_kv".to_string(), "deltanet_recurrent".to_string()];
        let base = compute_qwen35_prefix_hash(5, Some("q8"), &kinds, "plain", 0, &[1, 2, 3]);
        let same = compute_qwen35_prefix_hash(5, Some("q8"), &reordered, "plain", 0, &[1, 2, 3]);
        let different_tokens =
            compute_qwen35_prefix_hash(5, Some("q8"), &kinds, "plain", 0, &[1, 2, 4]);
        let different_prompt =
            compute_qwen35_prefix_hash(5, Some("q8"), &kinds, "open_think", 0, &[1, 2, 3]);
        assert_eq!(base, same);
        assert_ne!(base, different_tokens);
        assert_ne!(base, different_prompt);
        assert_eq!(base.algorithm, "xxh128");
        assert_eq!(base.value.len(), 32);
        assert!(base.value.bytes().all(|b| b.is_ascii_hexdigit()));
    }

    #[test]
    fn prefix_hash_json_shape_is_stable() {
        let hash = GenerateBatchPrefillPrefixHash {
            algorithm: "xxh128".to_string(),
            value: "0123456789abcdef0123456789abcdef".to_string(),
            prefix_len: 7,
        };
        assert_eq!(
            generate_prefix_hash_json(&hash),
            serde_json::json!({
                "algorithm": "xxh128",
                "value": "0123456789abcdef0123456789abcdef",
                "prefix_len": 7
            })
        );
    }

    #[test]
    fn semantic_boundary_checkpoint_json_shape_is_stable() {
        let checkpoint = Qwen35SemanticBoundaryCheckpoint {
            checkpoint_id: Some("qwen35-checkpoint:batch:req-1:7".to_string()),
            prefix_len: 7,
            hash: GenerateBatchPrefillPrefixHash {
                algorithm: "xxh128".to_string(),
                value: "0123456789abcdef0123456789abcdef".to_string(),
                prefix_len: 7,
            },
            boundary: "message_end".to_string(),
            boundary_index: 1,
        };
        assert_eq!(
            serde_json::to_value(checkpoint).unwrap(),
            serde_json::json!({
                "checkpoint_id": "qwen35-checkpoint:batch:req-1:7",
                "prefix_len": 7,
                "hash": {
                    "algorithm": "xxh128",
                    "value": "0123456789abcdef0123456789abcdef",
                    "prefix_len": 7
                },
                "boundary": "message_end",
                "boundary_index": 1
            })
        );
    }

    #[test]
    fn qwen35_prefill_checkpoint_hook_preserves_handle_contract() {
        let hash = GenerateBatchPrefillPrefixHash {
            algorithm: "xxh128".to_string(),
            value: "0123456789abcdef0123456789abcdef".to_string(),
            prefix_len: 12,
        };
        let final_hook = Qwen35PrefillCheckpointHook {
            batch_id: "batch-a",
            session_id: "req-1",
            source_state_handle: "req-1",
            logical_position: 12,
            kind: Qwen35PrefillCheckpointKind::Final,
            prefix_hash: &hash,
        };

        assert_eq!(final_hook.source_state_handle, "req-1");
        assert_eq!(final_hook.prefix_hash.prefix_len, 12);
        assert_eq!(qwen35_prefill_checkpoint_boundary_kind(final_hook), "full");
        assert_eq!(
            qwen35_prefill_checkpoint_session_id(final_hook),
            "qwen35-checkpoint:batch-a:req-1:12"
        );

        let boundary_hook = Qwen35PrefillCheckpointHook {
            logical_position: 8,
            kind: Qwen35PrefillCheckpointKind::SemanticBoundary {
                boundary: "message_end",
                boundary_index: 3,
            },
            ..final_hook
        };

        assert_eq!(
            qwen35_prefill_checkpoint_boundary_kind(boundary_hook),
            "message_end"
        );
        assert_eq!(
            qwen35_prefill_checkpoint_session_id(boundary_hook),
            "qwen35-checkpoint:batch-a:req-1:boundary:3:8"
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
    fn qwen35_prefill_backend_selection_matches_daemon_policy() {
        assert_eq!(
            select_qwen35_prefill_batch_backend(
                GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
                None,
                Ok(())
            )
            .unwrap(),
            Qwen35PrefillBatchBackend::FusedDense
        );
        assert_eq!(
            select_qwen35_prefill_batch_backend(
                GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
                None,
                Err("capability missing".to_string())
            )
            .unwrap(),
            Qwen35PrefillBatchBackend::SerialReference
        );
        assert_eq!(
            select_qwen35_prefill_batch_backend(
                GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
                Some("fused"),
                Ok(())
            )
            .unwrap(),
            Qwen35PrefillBatchBackend::FusedGroupedMoe
        );
        assert!(select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::SerialExact,
            Some("fused_grouped_moe"),
            Ok(())
        )
        .unwrap_err()
        .contains("grouped-MoE eligible"));
    }

    #[test]
    fn qwen35_prefill_scratch_target_batch_matches_daemon_policy() {
        assert_eq!(qwen35_prefill_scratch_target_batch(true, 16, None, 256), 16);
        assert_eq!(qwen35_prefill_scratch_target_batch(true, 1, None, 256), 2);
        assert_eq!(
            qwen35_prefill_scratch_target_batch(true, 16, Some("64"), 256),
            64
        );
        assert_eq!(
            qwen35_prefill_scratch_target_batch(false, 16, None, 256),
            256
        );
        assert_eq!(
            qwen35_prefill_scratch_target_batch(false, 300, None, 256),
            300
        );
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
