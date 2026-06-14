use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Outbound (CLI → daemon)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Inbound (daemon → CLI)
// ---------------------------------------------------------------------------

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
