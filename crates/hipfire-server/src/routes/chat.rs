use std::convert::Infallible;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::model::discovery::find_model;
use crate::state::SharedState;
use hipfire_config::HipfireConfig;
use hipfire_daemon_adapter::{
    find_daemon_bin_or_error, DaemonEngine, GenerateStreamControl, GenerateStreamEvent,
};
use hipfire_generate::{
    openai_chat_completion_done_chunk_json, GenerateTextRequest, GenerationSamplingPolicy,
};
use hipfire_model::{discover_dflash_draft_for_model, ModelLoadParams, ModelWorkerKey};
use hipfire_prompt::{Message as PromptMessage, Role, ToolCall as PromptToolCall};
use hipfire_scheduler::{
    create_request_session_draft, server_prefill_batch_enabled, CreateRequestSessionInput,
    NextBatchInput, SchedulerPolicyEnv,
};

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub max_tokens: Option<u32>,
    pub stop: Option<Value>,
    pub priority: Option<i64>,
    pub tools: Option<Value>,
    pub system: Option<String>,
    pub reasoning_effort: Option<String>,
    pub reasoning: Option<Value>,
    pub stream_options: Option<StreamOptions>,
    pub chat_template_kwargs: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
}

#[derive(Debug, Default, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<Value>,
    #[serde(default)]
    pub tool_calls: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

pub async fn post_chat_completions(
    State(state): State<SharedState>,
    Json(body): Json<ChatRequest>,
) -> Response {
    maybe_dump_request(&body);
    if body.stream {
        stream_chat(state, body).await.into_response()
    } else {
        blocking_chat(state, body).await
    }
}

fn maybe_dump_request(body: &ChatRequest) {
    if std::env::var("HIPFIRE_DUMP_REQUEST").ok().as_deref() != Some("1") {
        return;
    }
    let path = format!("/tmp/hipfire-request-{}.json", now_ms());
    match serde_json::to_string_pretty(body) {
        Ok(payload) => match std::fs::write(&path, payload) {
            Ok(()) => {
                let tool_count = match body.tools.as_ref() {
                    Some(Value::Array(items)) => items.len(),
                    Some(Value::Null) | None => 0,
                    Some(_) => 1,
                };
                tracing::error!(
                    path = %path,
                    messages = body.messages.len(),
                    tools = tool_count,
                    stream = body.stream,
                    "dumped request"
                );
            }
            Err(e) => tracing::error!(error = %e, path = %path, "request dump failed"),
        },
        Err(e) => tracing::error!(error = %e, "request dump serialization failed"),
    }
}

fn load_params_from_config(cfg: &HipfireConfig) -> ModelLoadParams {
    ModelLoadParams::from_hipfire_config(cfg)
}

#[derive(Debug, Clone, Copy)]
struct DaemonSpawnEnv {
    prompt_normalize: bool,
    dflash_ngram_block: bool,
}

impl DaemonSpawnEnv {
    fn from_resolved_config(cfg: &HipfireConfig, model_arg: &str) -> Self {
        Self {
            prompt_normalize: cfg.prompt_normalize,
            dflash_ngram_block: resolve_dflash_ngram_block(&cfg.dflash_ngram_block, model_arg),
        }
    }

    fn apply(self) {
        std::env::set_var(
            "HIPFIRE_NORMALIZE_PROMPT",
            if self.prompt_normalize { "1" } else { "0" },
        );
        if self.dflash_ngram_block {
            std::env::set_var("HIPFIRE_DFLASH_NGRAM_BLOCK", "1");
        } else {
            std::env::remove_var("HIPFIRE_DFLASH_NGRAM_BLOCK");
        }
    }
}

fn resolve_dflash_ngram_block(value: &Value, model_arg: &str) -> bool {
    if let Some(flag) = value.as_bool() {
        return flag;
    }
    if value.as_str() != Some("auto") {
        return false;
    }
    let model = model_arg.to_ascii_lowercase();
    [
        ":0.6b", "-0.6b", ":0.8b", "-0.8b", ":1b", "-1b", ":2b", "-2b", ":4b", "-4b",
    ]
    .iter()
    .any(|needle| model.contains(needle))
}

fn load_params_for_model_config(
    cfg: &HipfireConfig,
    model_arg: &str,
    model_path: Option<&Path>,
) -> ModelLoadParams {
    let resolved = cfg.resolve_for_model(model_arg);
    let mut params = load_params_from_config(&resolved);
    let min_viable = resolved.max_tokens.saturating_add(1024);
    if params.max_seq < min_viable {
        params.max_seq = min_viable;
    }
    maybe_attach_dflash_draft(&resolved, model_path, &mut params);
    maybe_attach_cask_sidecar(&resolved, model_path, &mut params);
    params
}

fn maybe_attach_dflash_draft(
    cfg: &HipfireConfig,
    model_path: Option<&Path>,
    params: &mut ModelLoadParams,
) {
    if cfg.dflash_mode == "off" {
        return;
    }
    let is_a3b = model_path
        .and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .map(|name| name.to_ascii_lowercase().contains("a3b"))
        .unwrap_or(false);
    let has_explicit_sidecar = cfg
        .cask_sidecar
        .as_deref()
        .filter(|sidecar| !sidecar.is_empty())
        .is_some_and(|sidecar| Path::new(sidecar).is_file());
    let allowed =
        cfg.dflash_mode == "on" || (cfg.dflash_mode == "auto" && (!is_a3b || has_explicit_sidecar));
    if !allowed {
        return;
    }
    if let Ok(explicit) = std::env::var("HIPFIRE_DFLASH_DRAFT") {
        if !explicit.is_empty() {
            params.draft = Some(explicit);
        }
        return;
    }
    if let Some(path) = model_path.and_then(discover_dflash_draft_for_model) {
        params.draft = Some(path.to_string_lossy().into_owned());
    }
}

fn maybe_attach_cask_sidecar(
    cfg: &HipfireConfig,
    model_path: Option<&Path>,
    params: &mut ModelLoadParams,
) {
    if std::env::var("HIPFIRE_CASK_OFF").ok().as_deref() == Some("1") {
        params.cask_sidecar = None;
        params.cask = None;
        params.cask_budget = None;
        params.cask_beta = None;
        params.cask_core_frac = None;
        params.cask_fold_m = None;
        return;
    }

    if let Some(sidecar) = cfg.cask_sidecar.as_deref().filter(|s| !s.is_empty()) {
        if Path::new(sidecar).is_file() {
            params.cask_sidecar = Some(sidecar.to_string());
            attach_cask_policy(cfg, params);
        } else {
            params.cask_sidecar = None;
            clear_cask_policy(params);
        }
        return;
    }

    if !cfg.cask_auto_attach {
        return;
    }
    let Some(model_path) = model_path else {
        return;
    };
    let filename = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if filename.to_ascii_lowercase().contains("a3b") {
        return;
    }
    if let Some(sidecar) = discover_triattn_sidecar(model_path) {
        params.cask_sidecar = Some(sidecar.to_string_lossy().into_owned());
        attach_cask_policy(cfg, params);
    }
}

fn attach_cask_policy(cfg: &HipfireConfig, params: &mut ModelLoadParams) {
    params.cask = Some(cfg.cask);
    params.cask_budget = Some(cfg.cask_budget);
    params.cask_beta = Some(cfg.cask_beta);
    params.cask_core_frac = Some(cfg.cask_core_frac);
    params.cask_fold_m = Some(cfg.cask_fold_m);
}

fn clear_cask_policy(params: &mut ModelLoadParams) {
    params.cask = None;
    params.cask_budget = None;
    params.cask_beta = None;
    params.cask_core_frac = None;
    params.cask_fold_m = None;
}

fn discover_triattn_sidecar(model_path: &Path) -> Option<std::path::PathBuf> {
    let filename = model_path.file_name().and_then(|name| name.to_str())?;
    let model_dir = model_path.parent().unwrap_or_else(|| Path::new("."));
    let mut dirs = vec![
        model_dir.to_path_buf(),
        hipfire_config::hipfire_dir().join("triattn"),
    ];
    dirs.dedup();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with(&format!("{filename}.triattn"))
                && (name.ends_with(".bin") || name.ends_with(".hfq"))
            {
                return Some(path);
            }
        }
    }
    None
}

#[derive(Clone, Debug)]
pub(crate) struct LoadedModelContext {
    pub(crate) model_path: String,
    pub(crate) worker_key_id: Option<String>,
    pub(crate) cache_capable: bool,
}

const MAX_REQUEST_TOKENS: u32 = 131_072;
const MAX_LOAD_MAX_SEQ: u32 = 524_288;

pub(crate) struct BlockingChatResult {
    pub(crate) req_id: String,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) text: String,
    pub(crate) done: hipfire_generate::DoneEvent,
    pub(crate) tool_calls: Vec<Value>,
    pub(crate) request_max_tokens: u32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RequestGenerationControls {
    pub(crate) presence_penalty: Option<f64>,
    pub(crate) frequency_penalty: Option<f64>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) thinking_mode: Option<String>,
    pub(crate) assistant_prefix: Option<String>,
    pub(crate) max_think_tokens: Option<u32>,
}

pub(crate) fn generate_request_from_chat(
    id: String,
    messages: &[ChatMessage],
    sampling: GenerationSamplingPolicy,
    worker_key_id: Option<String>,
    tools: Option<Value>,
    system: Option<String>,
    stop: Option<Vec<String>>,
    image_base64: Option<String>,
    controls: RequestGenerationControls,
) -> GenerateTextRequest {
    let mut req = GenerateTextRequest::from_openai_chat_messages(
        id,
        messages
            .iter()
            .map(|message| (message.role.as_str(), message.content.as_ref())),
        sampling,
    );
    req.messages = Some(chat_messages_to_prompt_messages(messages));
    req.with_worker_key_id(worker_key_id)
        .with_tools(tools)
        .with_system(system)
        .with_stop(stop)
        .with_image_base64(image_base64)
        .with_penalties(controls.presence_penalty, controls.frequency_penalty)
        .with_thinking_controls(
            controls.reasoning_effort,
            controls.thinking_mode,
            controls.assistant_prefix,
            controls.max_think_tokens,
        )
}

fn chat_messages_to_prompt_messages(messages: &[ChatMessage]) -> Vec<PromptMessage> {
    messages
        .iter()
        .filter_map(chat_message_to_prompt_message)
        .collect()
}

fn chat_message_to_prompt_message(message: &ChatMessage) -> Option<PromptMessage> {
    let role = chat_role_to_prompt_role(&message.role)?;
    let mut content = openai_chat_content_to_text(message.content.as_ref());
    if matches!(role, Role::Assistant) {
        content = strip_visible_thinking(content, false, false);
    }
    Some(PromptMessage {
        role,
        content,
        tool_calls: if matches!(role, Role::Assistant) {
            parse_openai_tool_calls(message.tool_calls.as_deref())
        } else {
            Vec::new()
        },
        tool_call_id: matches!(role, Role::Tool)
            .then(|| message.tool_call_id.clone())
            .flatten(),
    })
}

fn chat_role_to_prompt_role(role: &str) -> Option<Role> {
    match role {
        "developer" | "system" => Some(Role::System),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        "tool" | "toolResult" | "tool_result" => Some(Role::Tool),
        _ => None,
    }
}

fn openai_chat_content_to_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    part.get("text").and_then(Value::as_str)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

fn parse_openai_tool_calls(calls: Option<&[Value]>) -> Vec<PromptToolCall> {
    calls
        .unwrap_or(&[])
        .iter()
        .filter_map(|call| {
            let function = call.get("function").unwrap_or(call);
            let name = function.get("name").and_then(Value::as_str)?.to_string();
            let arguments = match function.get("arguments") {
                Some(Value::String(s)) => {
                    serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({"_raw": s}))
                }
                Some(value) => value.clone(),
                None => json!({}),
            };
            Some(PromptToolCall { name, arguments })
        })
        .collect()
}

pub(crate) async fn ensure_model_loaded(
    state: &SharedState,
    model_arg: &str,
    required_max_seq: u32,
) -> Result<LoadedModelContext, String> {
    let model_path =
        find_model(model_arg).ok_or_else(|| format!("model not found: {model_arg}"))?;
    let model_str = model_path.to_string_lossy().into_owned();

    let (params, daemon_spawn_env) = {
        let cfg = state.config.lock().await;
        let resolved_cfg = cfg.resolve_for_model(model_arg);
        let mut params = load_params_for_model_config(&cfg, model_arg, Some(&model_path));
        if required_max_seq > params.max_seq {
            params.max_seq = required_max_seq;
        }
        (
            params,
            DaemonSpawnEnv::from_resolved_config(&resolved_cfg, model_arg),
        )
    };

    let mut engine_guard = state.engine.lock().await;
    let mut loaded_guard = state.loaded_model_path.lock().await;

    if loaded_guard.as_deref() == Some(&model_str) {
        if let Some(eng) = engine_guard.as_mut() {
            if eng.ping().await.is_ok() {
                let cache_capable = state
                    .loaded_model_cache_capable
                    .lock()
                    .await
                    .unwrap_or(false);
                let loaded_max_seq = state.loaded_model_max_seq.lock().await.unwrap_or(0);
                if loaded_max_seq >= params.max_seq {
                    return Ok(LoadedModelContext {
                        model_path: model_str,
                        worker_key_id: eng.worker_key_id.clone(),
                        cache_capable,
                    });
                }
                tracing::info!(
                    model = %model_arg,
                    loaded_max_seq,
                    required_max_seq = params.max_seq,
                    "reloading model with larger max_seq for request"
                );
                eng.unload().await.map_err(|e| e.to_string())?;
                *loaded_guard = None;
                *state.loaded_model_cache_capable.lock().await = None;
                *state.loaded_model_max_seq.lock().await = None;
            }
        }
    }

    if let Some(eng) = engine_guard.as_mut() {
        if eng.ping().await.is_ok() {
            if loaded_guard.is_some() {
                eng.unload().await.map_err(|e| e.to_string())?;
                *loaded_guard = None;
                *state.loaded_model_cache_capable.lock().await = None;
                *state.loaded_model_max_seq.lock().await = None;
            }
            let loaded = eng
                .load(&model_str, params.clone())
                .await
                .map_err(|e| e.to_string())?;
            let cache_capable = loaded_response_cache_capable(&loaded);
            let worker_key_id = Some(loaded.worker_key_id);
            *loaded_guard = Some(model_str.clone());
            *state.loaded_model_cache_capable.lock().await = Some(cache_capable);
            *state.loaded_model_max_seq.lock().await = Some(params.max_seq);
            return Ok(LoadedModelContext {
                model_path: model_str,
                worker_key_id,
                cache_capable,
            });
        }
    }

    let bin = find_daemon_bin_or_error().map_err(|e| e.to_string())?;
    daemon_spawn_env.apply();
    let mut engine = DaemonEngine::spawn(&bin).await.map_err(|e| e.to_string())?;
    let loaded = engine
        .load(&model_str, params.clone())
        .await
        .map_err(|e| e.to_string())?;

    let cache_capable = loaded_response_cache_capable(&loaded);
    let worker_key_id = Some(loaded.worker_key_id);
    *loaded_guard = Some(model_str);
    *state.loaded_model_cache_capable.lock().await = Some(cache_capable);
    *state.loaded_model_max_seq.lock().await = Some(params.max_seq);
    *engine_guard = Some(engine);
    Ok(LoadedModelContext {
        model_path: loaded_guard
            .as_ref()
            .expect("loaded model path set")
            .clone(),
        worker_key_id,
        cache_capable,
    })
}

fn loaded_response_cache_capable(loaded: &hipfire_model::ModelLoadedResponse) -> bool {
    loaded.cache_capable.unwrap_or_else(|| {
        matches!(
            loaded.arch.as_deref(),
            Some("qwen3_5" | "qwen3_5_moe" | "deepseek4")
        )
    })
}

pub(crate) fn effective_request_max_tokens(default_max_tokens: u32, requested: Option<u32>) -> u32 {
    match requested {
        Some(value) if (1..=MAX_REQUEST_TOKENS).contains(&value) => value,
        _ => default_max_tokens,
    }
}

pub(crate) fn required_load_max_seq(
    default_max_seq: u32,
    request_max_tokens: u32,
    has_image: bool,
) -> u32 {
    let visual_headroom = if has_image { 1024_u64 } else { 0 };
    let required = u64::from(request_max_tokens) + 1024 + visual_headroom;
    u64::from(default_max_seq)
        .max(required)
        .min(u64::from(MAX_LOAD_MAX_SEQ)) as u32
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn preserve_thinking(body: &ChatRequest) -> bool {
    body.chat_template_kwargs
        .as_ref()
        .and_then(|v| v.get("preserve_thinking"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn include_stream_usage(body: &ChatRequest) -> bool {
    body.stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false)
}

pub(crate) fn request_generation_controls(
    cfg: &HipfireConfig,
    chat_template_kwargs: Option<&Value>,
    reasoning_effort: Option<&str>,
    reasoning: Option<&Value>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
) -> RequestGenerationControls {
    let enable_thinking = chat_template_kwargs
        .and_then(|v| v.get("enable_thinking"))
        .and_then(Value::as_bool);
    let nested_effort = reasoning
        .and_then(|v| v.get("effort"))
        .and_then(Value::as_str);
    let effort = reasoning_effort.or(nested_effort);
    let effort_budget = effort.and_then(reasoning_effort_budget);

    let mut max_think_tokens = None;
    if cfg.thinking == "off" || enable_thinking == Some(false) {
        max_think_tokens = Some(1);
    }
    if let Some(budget) = effort_budget {
        max_think_tokens = (budget > 0).then_some(budget);
    }

    let thinking_disabled =
        cfg.thinking == "off" || enable_thinking == Some(false) || effort == Some("none");
    let assistant_prefix = if thinking_disabled {
        Some("closed_think".to_string())
    } else {
        Some("open_think".to_string())
    };
    let thinking_mode = if thinking_disabled {
        Some("chat".to_string())
    } else if matches!(effort, Some("max" | "xhigh")) {
        Some("max".to_string())
    } else {
        Some("thinking".to_string())
    };

    RequestGenerationControls {
        presence_penalty: Some(presence_penalty.unwrap_or(0.0).max(0.0)),
        frequency_penalty: Some(frequency_penalty.unwrap_or(0.0).max(0.0)),
        reasoning_effort: effort.map(str::to_string),
        thinking_mode,
        assistant_prefix,
        max_think_tokens,
    }
}

fn reasoning_effort_budget(effort: &str) -> Option<u32> {
    match effort {
        "none" => Some(1),
        "minimal" => Some(64),
        "low" => Some(256),
        "medium" => Some(1024),
        "high" => Some(4096),
        "xhigh" => Some(0),
        _ => None,
    }
}

pub(crate) fn normalize_stop_sequences(
    stop: Option<&Value>,
) -> Result<Option<Vec<String>>, String> {
    let Some(stop) = stop else {
        return Ok(None);
    };
    let mut out = match stop {
        Value::String(s) => vec![s.clone()],
        Value::Array(items) => {
            let mut seqs = Vec::new();
            for item in items {
                let Some(s) = item.as_str() else {
                    return Err("stop must be a string or array of strings".to_string());
                };
                seqs.push(s.to_string());
            }
            seqs
        }
        Value::Null => Vec::new(),
        _ => return Err("stop must be a string or array of strings".to_string()),
    };
    out.retain(|s| !s.is_empty());
    out.truncate(4);
    for seq in &mut out {
        if seq.len() > 64 {
            seq.truncate(64);
        }
    }
    Ok((!out.is_empty()).then_some(out))
}

fn extract_request_image_base64(messages: &[ChatMessage]) -> Result<Option<String>, String> {
    let last_user_index = messages.iter().rposition(|m| m.role == "user");
    let mut images = Vec::new();

    for (idx, message) in messages.iter().enumerate() {
        if message.role != "user" {
            continue;
        }
        let Some(Value::Array(parts)) = message.content.as_ref() else {
            continue;
        };
        for part in parts {
            if part.get("type").and_then(Value::as_str) != Some("image_url") {
                continue;
            }
            let Some(url) = part
                .get("image_url")
                .and_then(|image_url| image_url.get("url"))
                .and_then(Value::as_str)
            else {
                return Err("malformed image part - image_url.url is required".to_string());
            };
            if !url.starts_with("data:") {
                return Err("remote image URLs are not supported - embed images as base64 data: URLs (supported formats: png, jpeg)".to_string());
            }
            let raw = if let Some(raw) = url.strip_prefix("data:image/png;base64,") {
                raw
            } else if let Some(raw) = url.strip_prefix("data:image/jpeg;base64,") {
                raw
            } else {
                return Err("unsupported image format - supported: png, jpeg".to_string());
            };
            if Some(idx) != last_user_index {
                return Err(
                    "images in earlier user turns are not supported - image must be in the last user message"
                        .to_string(),
                );
            }
            images.push(raw.to_string());
            if images.len() > 1 {
                return Err(
                    "multiple images not supported - only one image per request".to_string()
                );
            }
        }
    }

    Ok(images.pop())
}

fn openai_usage_json(done: &hipfire_generate::DoneEvent) -> Value {
    let prefill_tokens = done.prefill_tokens.unwrap_or(0) as u64;
    let cached_tokens = done_extra_u64(done, "cached_tokens").unwrap_or(0);
    let prompt_tokens =
        done_extra_u64(done, "prompt_tokens").unwrap_or(cached_tokens + prefill_tokens);
    let completion_tokens = done.tokens as u64;
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": cached_tokens,
            "cache_write_tokens": prefill_tokens,
        },
        "cache_read_input_tokens": cached_tokens,
        "cache_creation_input_tokens": prefill_tokens,
    })
}

fn openai_nonstream_usage_json(done: &hipfire_generate::DoneEvent) -> Value {
    let prefill_tokens = done.prefill_tokens.unwrap_or(0) as u64;
    let cached_tokens = done_extra_u64(done, "cached_tokens").unwrap_or(0);
    let prompt_tokens =
        done_extra_u64(done, "prompt_tokens").unwrap_or(cached_tokens + prefill_tokens);
    let completion_tokens = done.tokens as u64;
    let cache_write_tokens = prompt_tokens.saturating_sub(cached_tokens);
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
        },
        "cache_read_input_tokens": cached_tokens,
        "cache_creation_input_tokens": cache_write_tokens,
    })
}

fn done_extra_u64(done: &hipfire_generate::DoneEvent, key: &str) -> Option<u64> {
    done.extra.get(key).and_then(Value::as_u64)
}

fn openai_timings_json(done: &hipfire_generate::DoneEvent) -> Value {
    let mut timings = serde_json::Map::new();
    timings.insert("tokens".to_string(), json!(done.tokens));
    if let Some(value) = done.tok_s {
        timings.insert("tok_s".to_string(), json!(value));
    }
    if let Some(value) = done.prefill_tokens {
        timings.insert("prefill_tokens".to_string(), json!(value));
    }
    if let Some(value) = done.prefill_ms {
        timings.insert("prefill_ms".to_string(), json!(value));
    }
    if let Some(value) = done.prefill_tok_s {
        timings.insert("prefill_tok_s".to_string(), json!(value));
    }
    if let Some(value) = done.decode_tok_s {
        timings.insert("decode_tok_s".to_string(), json!(value));
    }
    if let Some(value) = done.ttft_ms {
        timings.insert("ttft_ms".to_string(), json!(value));
    }
    for key in ["tau", "cycles", "dflash"] {
        if let Some(value) = done.extra.get(key) {
            timings.insert(key.to_string(), value.clone());
        }
    }
    Value::Object(timings)
}

pub(crate) fn openai_chat_completion_response_with_tool_calls_json(
    req_id: &str,
    created: u64,
    model: &str,
    text: &str,
    tool_calls: &[Value],
    done: &hipfire_generate::DoneEvent,
    request_max_tokens: u32,
) -> Value {
    let mut message = json!({
        "role": "assistant",
        "content": if text.is_empty() { Value::Null } else { Value::String(text.to_string()) },
    });
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls.to_vec());
    }
    let mut choice = json!({
        "index": 0,
        "message": message,
        "finish_reason": if tool_calls.is_empty() {
            done.finish_reason.as_deref().unwrap_or("stop")
        } else {
            "tool_calls"
        },
    });
    if tool_calls.is_empty() && text.is_empty() {
        choice["message"]["content"] = Value::String(String::new());
    }
    let mut response = json!({
        "id": format!("chatcmpl-{req_id}"),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [choice],
        "usage": openai_nonstream_usage_json(done)
    });
    if tool_calls.is_empty() {
        if let Some(truncation) = detect_tool_call_truncation(text, done.tokens, request_max_tokens)
        {
            response["choices"][0]["finish_reason"] = json!("length");
            response["truncation"] = truncation;
        }
    }
    response
}

fn daemon_tool_calls_to_openai(calls: Vec<hipfire_generate::ToolCall>, req_id: &str) -> Vec<Value> {
    calls
        .into_iter()
        .enumerate()
        .map(|(idx, call)| {
            json!({
                "id": format!("call_{req_id}_{idx}"),
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string()),
                }
            })
        })
        .collect()
}

fn parse_inline_tool_calls(text: &str, req_id: &str) -> (String, Vec<Value>) {
    let mut tool_calls = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel_open) = text[search_from..].find("<tool_call>") {
        let open = search_from + rel_open;
        let body_start = open + "<tool_call>".len();
        let Some(rel_close) = text[body_start..].find("</tool_call>") else {
            break;
        };
        let close = body_start + rel_close;
        let mut raw = text[body_start..close].trim();
        while let Some(stripped) = raw.strip_prefix("<tool_call>") {
            raw = stripped.trim_start();
        }
        if let Some(last_open) = raw.rfind("<tool_call>") {
            raw = raw[last_open + "<tool_call>".len()..].trim_start();
            if let Some((before_close, _)) = raw.split_once("</tool_call>") {
                raw = before_close.trim();
            }
        }
        if let Some((name, arguments)) = parse_one_inline_tool_call(raw) {
            let idx = tool_calls.len();
            tool_calls.push(json!({
                "id": format!("call_{req_id}_{idx}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string()),
                }
            }));
        }
        search_from = close + "</tool_call>".len();
    }

    if tool_calls.is_empty() {
        return (text.to_string(), tool_calls);
    }
    let content = text
        .split_once("<tool_call>")
        .map(|(before, _)| before.trim().to_string())
        .unwrap_or_default();
    (content, tool_calls)
}

fn parse_one_inline_tool_call(raw: &str) -> Option<(String, Value)> {
    let cleaned = raw
        .replace("<|im_start|>", "")
        .replace("<|im_end|>", "")
        .replace("<|endoftext|>", "")
        .replace("<|im_sep|>", "");
    let raw = cleaned.trim();
    if let Ok(value) = serde_json::from_str::<Value>(raw) {
        if let Some(name) = value.get("name").and_then(Value::as_str) {
            let arguments = value.get("arguments").cloned().unwrap_or_else(|| {
                let mut args = serde_json::Map::new();
                if let Some(obj) = value.as_object() {
                    for (key, val) in obj {
                        if !matches!(key.as_str(), "name" | "type" | "id" | "function") {
                            args.insert(key.clone(), val.clone());
                        }
                    }
                }
                Value::Object(args)
            });
            return Some((name.to_string(), arguments));
        }
    }

    if let Some((name, tail)) = inline_xml_tool_name_and_tail(raw) {
        let arguments = extract_first_json_object(tail).unwrap_or_else(|| json!({}));
        return Some((name, arguments));
    }

    if let Some(rest) = raw.strip_prefix("<function=") {
        let (name, after_name) = rest.split_once('>')?;
        let mut args = serde_json::Map::new();
        let mut body = after_name;
        while let Some(start) = body.find("<parameter=") {
            let key_start = start + "<parameter=".len();
            let Some(key_end_rel) = body[key_start..].find('>') else {
                break;
            };
            let key_end = key_start + key_end_rel;
            let key = &body[key_start..key_end];
            let value_start = key_end + 1;
            let Some(value_end_rel) = body[value_start..].find("</parameter>") else {
                break;
            };
            let value_end = value_start + value_end_rel;
            args.insert(
                key.to_string(),
                coerce_inline_tool_param(body[value_start..value_end].trim()),
            );
            body = &body[value_end + "</parameter>".len()..];
        }
        return Some((name.trim().to_string(), Value::Object(args)));
    }

    if let Some(name) = extract_json_string_field(raw, "name") {
        let arguments = find_field_tail(raw, "arguments")
            .and_then(extract_first_json_object)
            .or_else(|| extract_first_json_object(raw))
            .unwrap_or_else(|| json!({}));
        return Some((name, arguments));
    }

    None
}

fn coerce_inline_tool_param(s: &str) -> Value {
    if s.is_empty() {
        return Value::String(String::new());
    }
    serde_json::from_str::<Value>(s).unwrap_or_else(|_| Value::String(s.to_string()))
}

fn inline_xml_tool_name_and_tail(raw: &str) -> Option<(String, &str)> {
    if let Some(rest) = raw.strip_prefix("<plain>") {
        let name_end = rest.find("</param>")?;
        let name = rest[..name_end].trim();
        if !name.is_empty() {
            return Some((name.to_string(), &rest[name_end + "</param>".len()..]));
        }
    }
    if let Some(rest) = raw.strip_prefix("<tool name=") {
        let end = rest.find('>')?;
        let name = rest[..end].trim().trim_matches('"');
        if !name.is_empty() {
            return Some((name.to_string(), &rest[end + 1..]));
        }
    }
    None
}

fn find_field_tail<'a>(raw: &'a str, field: &str) -> Option<&'a str> {
    for needle in [
        format!("\"{field}\""),
        format!("'{field}'"),
        field.to_string(),
    ] {
        let Some(start) = raw.find(&needle) else {
            continue;
        };
        let after_key = &raw[start + needle.len()..];
        let Some(colon) = after_key.find(':') else {
            continue;
        };
        return Some(after_key[colon + 1..].trim_start());
    }
    None
}

fn extract_json_string_field(raw: &str, field: &str) -> Option<String> {
    let tail = find_field_tail(raw, field)?;
    let mut chars = tail.chars();
    let quote = chars.next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let mut out = String::new();
    let mut escape = false;
    for ch in chars {
        if escape {
            out.push(ch);
            escape = false;
            continue;
        }
        if ch == '\\' {
            escape = true;
            continue;
        }
        if ch == quote {
            return (!out.is_empty()).then_some(out);
        }
        out.push(ch);
    }
    None
}

fn extract_first_json_object(raw: &str) -> Option<Value> {
    let start = raw.find('{')?;
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escape = false;
    for (rel_idx, ch) in raw[start..].char_indices() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            if ch == '\\' {
                escape = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            continue;
        }
        if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
            if depth == 0 {
                let end = start + rel_idx + ch.len_utf8();
                return serde_json::from_str::<Value>(&raw[start..end]).ok();
            }
        }
    }
    None
}

fn detect_tool_call_truncation(
    text: &str,
    decoded_tokens: u32,
    max_tokens_cap: u32,
) -> Option<Value> {
    let opens = text.matches("<tool_call>").count();
    let closes = text.matches("</tool_call>").count();
    if opens <= closes {
        return None;
    }
    if decoded_tokens + 4 < max_tokens_cap {
        return None;
    }
    Some(json!({
        "reason": "max_tokens_in_tool_call",
        "max_tokens_used": decoded_tokens,
        "suggested_max_tokens": (max_tokens_cap.saturating_mul(4)).clamp(4096, 32768),
    }))
}

pub(crate) fn strip_visible_thinking(
    mut content: String,
    preserve: bool,
    started_in_think: bool,
) -> String {
    if preserve {
        return content.replace("<|im_end|>", "").trim().to_string();
    }
    if started_in_think && !content.contains("<think>") && content.contains("</think>") {
        content = format!("<think>{content}");
    }

    let mut out = String::with_capacity(content.len());
    let mut rest = content.as_str();
    loop {
        let Some(open) = rest.find("<think>") else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..open]);
        let after_open = &rest[open + "<think>".len()..];
        let Some(close) = after_open.find("</think>") else {
            break;
        };
        rest = &after_open[close + "</think>".len()..];
    }
    out.replace("</think>", "")
        .replace("<|im_end|>", "")
        .trim()
        .to_string()
}

pub(crate) enum AssistantDelta {
    Content(String),
    Reasoning(String),
}

#[derive(Default)]
pub(crate) struct ThinkStreamFilter {
    in_think: bool,
    strip_next_leading_newline: bool,
}

impl ThinkStreamFilter {
    pub(crate) fn observe(&mut self, text: &str, preserve: bool) -> Vec<AssistantDelta> {
        if preserve {
            let text = text.replace("<|im_end|>", "");
            return (!text.is_empty())
                .then_some(AssistantDelta::Content(text))
                .into_iter()
                .collect();
        }

        let mut out = Vec::new();
        let mut rest = text.replace("<|im_end|>", "");
        loop {
            if self.in_think {
                if let Some(close) = rest.find("</think>") {
                    let reasoning = &rest[..close];
                    if !reasoning.is_empty() {
                        out.push(AssistantDelta::Reasoning(reasoning.to_string()));
                    }
                    rest = rest[close + "</think>".len()..].to_string();
                    self.in_think = false;
                    self.strip_next_leading_newline = true;
                    continue;
                }
                if !rest.is_empty() {
                    out.push(AssistantDelta::Reasoning(rest));
                }
                break;
            }

            if let Some(open) = rest.find("<think>") {
                let content = &rest[..open];
                if !content.is_empty() {
                    out.push(AssistantDelta::Content(content.to_string()));
                }
                rest = rest[open + "<think>".len()..].to_string();
                self.in_think = true;
                continue;
            }

            if self.strip_next_leading_newline {
                rest = rest.trim_start_matches('\n').to_string();
                self.strip_next_leading_newline = false;
            }
            rest = rest.replace("</think>", "");
            if !rest.is_empty() {
                out.push(AssistantDelta::Content(rest));
            }
            break;
        }
        out
    }
}

fn openai_chat_completion_delta_chunk_json(
    req_id: &str,
    created: u64,
    model: &str,
    field: &str,
    text: &str,
) -> Value {
    let mut delta = serde_json::Map::new();
    delta.insert("role".to_string(), json!("assistant"));
    delta.insert(field.to_string(), json!(text));
    json!({
        "id": format!("chatcmpl-{req_id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": null
        }]
    })
}

fn openai_chat_completion_tool_call_chunk_json(
    req_id: &str,
    created: u64,
    model: &str,
    index: usize,
    tool_call: &Value,
) -> Value {
    let mut call = tool_call.clone();
    if let Value::Object(map) = &mut call {
        map.insert("index".to_string(), json!(index));
    }
    json!({
        "id": format!("chatcmpl-{req_id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "tool_calls": [call]
            },
            "finish_reason": null
        }]
    })
}

fn openai_chat_completion_finish_chunk_json(
    req_id: &str,
    created: u64,
    model: &str,
    finish_reason: &str,
) -> Value {
    json!({
        "id": format!("chatcmpl-{req_id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }]
    })
}

fn tools_present(tools: Option<&Value>) -> bool {
    match tools {
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Null) | None => false,
        Some(_) => true,
    }
}

fn estimated_prompt_tokens(messages: &[ChatMessage]) -> Vec<u32> {
    let mut tokens = Vec::new();
    for message in messages {
        if let Some(content) = &message.content {
            let text = content
                .as_str()
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| content.to_string());
            tokens.extend(text.as_bytes().iter().map(|b| u32::from(*b)));
        }
    }
    if tokens.is_empty() {
        tokens.push(0);
    }
    tokens
}

fn scheduler_worker_key(model_path: &str, cfg: &HipfireConfig) -> ModelWorkerKey {
    ModelWorkerKey {
        artifact_path: model_path.to_string(),
        artifact_digest: None,
        arch_id: "unknown".to_string(),
        quant_family: "unknown".to_string(),
        state_mode: cfg.kv_cache.clone(),
        max_seq_bucket: cfg.max_seq as usize,
        accelerator_kind: Some("hip".to_string()),
        device_id: Some("0".to_string()),
        feature_flags: vec!["rust-server".to_string(), "prefill-queue".to_string()],
    }
}

pub(crate) async fn wait_for_prefill_scheduler_turn(
    state: &SharedState,
    req_id: &str,
    model_path: &str,
    messages: &[ChatMessage],
    priority: Option<i64>,
) -> Result<(), String> {
    let env = SchedulerPolicyEnv::from_pairs(std::env::vars());
    if !server_prefill_batch_enabled(&env) {
        return Ok(());
    }

    let worker_key = {
        let cfg = state.config.lock().await;
        scheduler_worker_key(model_path, &cfg)
    };
    let session = create_request_session_draft(CreateRequestSessionInput {
        id: req_id.to_string(),
        worker_key,
        prompt_tokens: estimated_prompt_tokens(messages),
        cached_prefix_tokens: None,
        priority,
        state_kinds: vec!["kv".to_string()],
    });

    {
        let mut scheduler = state.prefill_scheduler.lock().await;
        scheduler.enqueue(session, now_ms())?;
    }
    state.prefill_notify.notify_waiters();

    loop {
        {
            let mut selected = state.selected_prefill_requests.lock().await;
            if selected.remove(req_id) {
                return Ok(());
            }
        }

        {
            let _dispatch = state.prefill_dispatch.lock().await;
            {
                let mut selected = state.selected_prefill_requests.lock().await;
                if selected.remove(req_id) {
                    return Ok(());
                }
            }

            let batch = {
                let mut scheduler = state.prefill_scheduler.lock().await;
                scheduler.next_prefill_batch(NextBatchInput { now_ms: now_ms() })
            };

            if let Some(batch) = batch {
                let mut selected = state.selected_prefill_requests.lock().await;
                for session in batch.sessions {
                    selected.insert(session.id);
                }
                state.prefill_notify.notify_waiters();
                continue;
            }
        }

        tokio::select! {
            _ = state.prefill_notify.notified() => {}
            _ = tokio::time::sleep(Duration::from_millis(2)) => {}
        }
    }
}

pub(crate) async fn execute_blocking_chat(
    state: SharedState,
    body: ChatRequest,
) -> Result<BlockingChatResult, Value> {
    match execute_blocking_chat_cancellable(state, body, || false).await? {
        Some(result) => Ok(result),
        None => Err(json!({"error": {"message": "request cancelled", "type": "server_error"}})),
    }
}

async fn execute_blocking_chat_cancellable<F>(
    state: SharedState,
    body: ChatRequest,
    mut should_cancel: F,
) -> Result<Option<BlockingChatResult>, Value>
where
    F: FnMut() -> bool,
{
    let req_id = Uuid::new_v4().to_string();
    let created = now_unix_seconds();

    let model_arg = {
        let cfg = state.config.lock().await;
        body.model.clone().or(cfg.default_model.clone())
    };

    let Some(model_arg) = model_arg else {
        return Err(
            json!({"error": {"message": "no model specified", "type": "invalid_request_error"}}),
        );
    };
    let preserve_thinking = preserve_thinking(&body);
    let stop = match normalize_stop_sequences(body.stop.as_ref()) {
        Ok(stop) => stop,
        Err(message) => {
            return Err(json!({"error": {"message": message, "type": "invalid_request_error"}}));
        }
    };
    let image_base64 = match extract_request_image_base64(&body.messages) {
        Ok(image_base64) => image_base64,
        Err(message) => {
            return Err(json!({"error": {"message": message, "type": "invalid_request_error"}}));
        }
    };
    let (request_max_tokens, required_max_seq) = {
        let cfg = state.config.lock().await;
        let resolved = cfg.resolve_for_model(&model_arg);
        let request_max_tokens = effective_request_max_tokens(resolved.max_tokens, body.max_tokens);
        (
            request_max_tokens,
            required_load_max_seq(resolved.max_seq, request_max_tokens, image_base64.is_some()),
        )
    };

    let loaded = match ensure_model_loaded(&state, &model_arg, required_max_seq).await {
        Ok(loaded) => loaded,
        Err(e) => return Err(json!({"error": {"message": e, "type": "server_error"}})),
    };

    if let Err(e) = wait_for_prefill_scheduler_turn(
        &state,
        &req_id,
        &loaded.model_path,
        &body.messages,
        body.priority,
    )
    .await
    {
        return Err(json!({"error": {"message": e, "type": "server_error"}}));
    }

    let gen_req = {
        let cfg = state.config.lock().await;
        let resolved = cfg.resolve_for_model(&model_arg);
        let controls = request_generation_controls(
            &resolved,
            body.chat_template_kwargs.as_ref(),
            body.reasoning_effort.as_deref(),
            body.reasoning.as_ref(),
            body.presence_penalty,
            body.frequency_penalty,
        );
        generate_request_from_chat(
            req_id.clone(),
            &body.messages,
            GenerationSamplingPolicy::from_defaults(
                resolved.temperature,
                resolved.top_p,
                resolved.repeat_penalty,
                resolved.max_tokens,
                body.temperature,
                body.top_p,
                body.repeat_penalty,
                Some(request_max_tokens),
            ),
            loaded.worker_key_id,
            body.tools,
            body.system,
            stop,
            image_base64,
            controls,
        )
    };

    let mut engine_guard = state.engine.lock().await;
    let mut engine = match engine_guard.take() {
        Some(e) => e,
        None => {
            return Err(json!({"error": {"message": "daemon not running", "type": "server_error"}}))
        }
    };

    if !loaded.cache_capable {
        if let Err(e) = engine.reset().await {
            *engine_guard = Some(engine);
            return Err(json!({"error": {"message": e.to_string(), "type": "server_error"}}));
        }
    }

    let mut raw_text = String::new();
    let mut raw_tool_calls = Vec::new();
    let result = engine
        .generate_streaming_events_controlled(gen_req, |event| {
            if should_cancel() {
                return GenerateStreamControl::Cancel;
            }
            match event {
                GenerateStreamEvent::Token(token) => raw_text.push_str(&token),
                GenerateStreamEvent::ToolCalls(calls) => raw_tool_calls.extend(calls),
            }
            GenerateStreamControl::Continue
        })
        .await;

    match result {
        Ok(Some(done)) => {
            *engine_guard = Some(engine);
            let mut text = strip_visible_thinking(raw_text, preserve_thinking, true);
            let mut tool_calls = daemon_tool_calls_to_openai(raw_tool_calls, &req_id);
            if tool_calls.is_empty() {
                let (content, parsed) = parse_inline_tool_calls(&text, &req_id);
                text = content;
                tool_calls = parsed;
            } else if let Some((before, _)) = text.split_once("<tool_call>") {
                text = before.trim().to_string();
            }
            Ok(Some(BlockingChatResult {
                req_id,
                created,
                model: model_arg,
                text,
                done,
                tool_calls,
                request_max_tokens,
            }))
        }
        Ok(None) => {
            tracing::info!(request_id = %req_id, "non-stream client disconnected; dropping daemon");
            *state.loaded_model_path.lock().await = None;
            *state.loaded_model_cache_capable.lock().await = None;
            *state.loaded_model_max_seq.lock().await = None;
            drop(engine);
            Ok(None)
        }
        Err(e) => {
            *engine_guard = Some(engine);
            Err(json!({"error": {"message": e.to_string(), "type": "server_error"}}))
        }
    }
}

fn blocking_chat_response_json(result: Result<BlockingChatResult, Value>) -> Value {
    match result {
        Ok(result) => openai_chat_completion_response_with_tool_calls_json(
            &result.req_id,
            result.created,
            &result.model,
            &result.text,
            &result.tool_calls,
            &result.done,
            result.request_max_tokens,
        ),
        Err(e) => e,
    }
}

async fn blocking_chat(state: SharedState, body: ChatRequest) -> Response {
    if let Some((status, error)) = blocking_chat_preflight_error(&state, &body).await {
        return (status, Json(error)).into_response();
    }

    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(1);
    tokio::spawn(async move {
        match execute_blocking_chat_cancellable(state, body, || tx.is_closed()).await {
            Ok(Some(result)) => {
                let payload = blocking_chat_response_json(Ok(result));
                if let Ok(bytes) = serde_json::to_vec(&payload) {
                    let _ = tx.send(bytes).await;
                }
            }
            Ok(None) => {}
            Err(error) => {
                let payload = blocking_chat_response_json(Err(error));
                if let Ok(bytes) = serde_json::to_vec(&payload) {
                    let _ = tx.send(bytes).await;
                }
            }
        }
    });

    let stream = async_stream::stream! {
        let mut heartbeat = tokio::time::interval(Duration::from_secs(10));
        loop {
            tokio::select! {
                result = rx.recv() => {
                    if let Some(bytes) = result {
                        yield Ok::<Bytes, Infallible>(Bytes::from(bytes));
                    }
                    break;
                }
                _ = heartbeat.tick() => {
                    yield Ok::<Bytes, Infallible>(Bytes::from_static(b" "));
                }
            }
        }
    };

    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| Response::new(Body::from("{}")))
}

async fn blocking_chat_preflight_error(
    state: &SharedState,
    body: &ChatRequest,
) -> Option<(StatusCode, Value)> {
    let model_arg = {
        let cfg = state.config.lock().await;
        body.model.clone().or(cfg.default_model.clone())
    };
    let Some(model_arg) = model_arg else {
        return Some((
            StatusCode::BAD_REQUEST,
            json!({"error": {"message": "no model specified", "type": "invalid_request_error"}}),
        ));
    };
    if let Err(message) = normalize_stop_sequences(body.stop.as_ref()) {
        return Some((
            StatusCode::BAD_REQUEST,
            json!({"error": {"message": message, "type": "invalid_request_error"}}),
        ));
    }
    if let Err(message) = extract_request_image_base64(&body.messages) {
        return Some((
            StatusCode::BAD_REQUEST,
            json!({"error": {"message": message, "type": "invalid_request_error"}}),
        ));
    }
    if find_model(&model_arg).is_none() {
        return Some((
            StatusCode::NOT_FOUND,
            json!({"error": {"message": format!("model not found: {model_arg}"), "type": "invalid_request_error"}}),
        ));
    }
    None
}

#[allow(dead_code)]
async fn blocking_chat_buffered_for_tests(state: SharedState, body: ChatRequest) -> Response {
    match execute_blocking_chat(state, body).await {
        Ok(result) => Json(openai_chat_completion_response_with_tool_calls_json(
            &result.req_id,
            result.created,
            &result.model,
            &result.text,
            &result.tool_calls,
            &result.done,
            result.request_max_tokens,
        ))
        .into_response(),
        Err(e) => {
            let status = if e
                .get("error")
                .and_then(|error| error.get("type"))
                .and_then(Value::as_str)
                == Some("invalid_request_error")
            {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(e)).into_response()
        }
    }
}

async fn stream_chat(state: SharedState, body: ChatRequest) -> impl IntoResponse {
    let (tx, mut rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        let req_id = Uuid::new_v4().to_string();
        let created = now_unix_seconds();

        let model_arg = {
            let cfg = state.config.lock().await;
            body.model.clone().or(cfg.default_model.clone())
        };

        let model_arg = match model_arg {
            Some(m) => m,
            None => {
                let ev = sse_error("no model specified");
                let _ = tx.send(Ok(ev)).await;
                let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                return;
            }
        };
        let preserve_thinking = preserve_thinking(&body);
        let include_usage = include_stream_usage(&body);
        let has_tools = tools_present(body.tools.as_ref());
        let stop = match normalize_stop_sequences(body.stop.as_ref()) {
            Ok(stop) => stop,
            Err(message) => {
                let _ = tx.send(Ok(sse_error(&message))).await;
                let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                return;
            }
        };
        let image_base64 = match extract_request_image_base64(&body.messages) {
            Ok(image_base64) => image_base64,
            Err(message) => {
                let _ = tx.send(Ok(sse_error(&message))).await;
                let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                return;
            }
        };
        let (request_max_tokens, required_max_seq) = {
            let cfg = state.config.lock().await;
            let resolved = cfg.resolve_for_model(&model_arg);
            let request_max_tokens =
                effective_request_max_tokens(resolved.max_tokens, body.max_tokens);
            (
                request_max_tokens,
                required_load_max_seq(resolved.max_seq, request_max_tokens, image_base64.is_some()),
            )
        };

        let loaded = match ensure_model_loaded(&state, &model_arg, required_max_seq).await {
            Ok(loaded) => loaded,
            Err(e) => {
                let _ = tx.send(Ok(sse_error(&e))).await;
                let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                return;
            }
        };

        if let Err(e) = wait_for_prefill_scheduler_turn(
            &state,
            &req_id,
            &loaded.model_path,
            &body.messages,
            body.priority,
        )
        .await
        {
            let _ = tx.send(Ok(sse_error(&e))).await;
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
            return;
        }

        let gen_req = {
            let cfg = state.config.lock().await;
            let resolved = cfg.resolve_for_model(&model_arg);
            let controls = request_generation_controls(
                &resolved,
                body.chat_template_kwargs.as_ref(),
                body.reasoning_effort.as_deref(),
                body.reasoning.as_ref(),
                body.presence_penalty,
                body.frequency_penalty,
            );
            generate_request_from_chat(
                req_id.clone(),
                &body.messages,
                GenerationSamplingPolicy::from_defaults(
                    resolved.temperature,
                    resolved.top_p,
                    resolved.repeat_penalty,
                    resolved.max_tokens,
                    body.temperature,
                    body.top_p,
                    body.repeat_penalty,
                    Some(request_max_tokens),
                ),
                loaded.worker_key_id,
                body.tools,
                body.system,
                stop,
                image_base64,
                controls,
            )
        };

        let req_id_cb = req_id.clone();
        let created_cb = created;
        let model_cb = model_arg.clone();
        let tx_cb = tx.clone();
        let mut think_filter = ThinkStreamFilter::default();
        let mut accumulated_tool_text = String::new();
        let mut structured_tool_calls_emitted = false;
        let mut next_tool_call_index = 0usize;

        let mut engine_guard = state.engine.lock().await;
        let mut engine = match engine_guard.take() {
            Some(e) => e,
            None => {
                let _ = tx.send(Ok(sse_error("daemon not running"))).await;
                let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                return;
            }
        };

        if !loaded.cache_capable {
            if let Err(e) = engine.reset().await {
                *engine_guard = Some(engine);
                let _ = tx.send(Ok(sse_error(&e.to_string()))).await;
                let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                return;
            }
        }

        let result = engine
            .generate_streaming_events_controlled(gen_req, |event| match event {
                GenerateStreamEvent::Token(token) => {
                    if tx_cb.is_closed() {
                        return GenerateStreamControl::Cancel;
                    }
                    if has_tools {
                        accumulated_tool_text.push_str(&token);
                        return GenerateStreamControl::Continue;
                    }
                    for delta in think_filter.observe(&token, preserve_thinking) {
                        let chunk = match delta {
                            AssistantDelta::Content(text) => {
                                openai_chat_completion_delta_chunk_json(
                                    &req_id_cb, created_cb, &model_cb, "content", &text,
                                )
                            }
                            AssistantDelta::Reasoning(text) => {
                                openai_chat_completion_delta_chunk_json(
                                    &req_id_cb,
                                    created_cb,
                                    &model_cb,
                                    "reasoning_content",
                                    &text,
                                )
                            }
                        };
                        match tx_cb.try_send(Ok(
                            Event::default().data(serde_json::to_string(&chunk).unwrap())
                        )) {
                            Ok(()) => {}
                            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                                return GenerateStreamControl::Cancel;
                            }
                            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {}
                        }
                    }
                    GenerateStreamControl::Continue
                }
                GenerateStreamEvent::ToolCalls(calls) => {
                    if tx_cb.is_closed() {
                        return GenerateStreamControl::Cancel;
                    }
                    let tool_calls = daemon_tool_calls_to_openai(calls, &req_id_cb);
                    for tool_call in tool_calls {
                        structured_tool_calls_emitted = true;
                        let chunk = openai_chat_completion_tool_call_chunk_json(
                            &req_id_cb,
                            created_cb,
                            &model_cb,
                            next_tool_call_index,
                            &tool_call,
                        );
                        next_tool_call_index += 1;
                        match tx_cb.try_send(Ok(
                            Event::default().data(serde_json::to_string(&chunk).unwrap())
                        )) {
                            Ok(()) => {}
                            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                                return GenerateStreamControl::Cancel;
                            }
                            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {}
                        }
                    }
                    GenerateStreamControl::Continue
                }
            })
            .await;

        match result {
            Ok(Some(done)) => {
                *engine_guard = Some(engine);
                let mut final_chunk = if has_tools {
                    if !structured_tool_calls_emitted {
                        let stripped =
                            strip_visible_thinking(accumulated_tool_text, preserve_thinking, true);
                        let (content, tool_calls) = parse_inline_tool_calls(&stripped, &req_id);
                        let truncation = if tool_calls.is_empty() {
                            detect_tool_call_truncation(&stripped, done.tokens, request_max_tokens)
                        } else {
                            None
                        };
                        if !content.is_empty() {
                            let content_chunk = openai_chat_completion_delta_chunk_json(
                                &req_id, created, &model_arg, "content", &content,
                            );
                            let _ = tx
                                .send(Ok(Event::default()
                                    .data(serde_json::to_string(&content_chunk).unwrap())))
                                .await;
                        }
                        for (idx, tool_call) in tool_calls.iter().enumerate() {
                            let chunk = openai_chat_completion_tool_call_chunk_json(
                                &req_id, created, &model_arg, idx, tool_call,
                            );
                            let _ = tx
                                .send(Ok(
                                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                                ))
                                .await;
                        }
                        if tool_calls.is_empty() {
                            let mut chunk =
                                openai_chat_completion_done_chunk_json(&req_id, &model_arg, &done);
                            chunk["created"] = json!(created);
                            if let Some(truncation) = truncation {
                                chunk["choices"][0]["finish_reason"] = json!("length");
                                chunk["truncation"] = truncation;
                            }
                            chunk
                        } else {
                            openai_chat_completion_finish_chunk_json(
                                &req_id,
                                created,
                                &model_arg,
                                "tool_calls",
                            )
                        }
                    } else {
                        openai_chat_completion_finish_chunk_json(
                            &req_id,
                            created,
                            &model_arg,
                            "tool_calls",
                        )
                    }
                } else {
                    let mut chunk =
                        openai_chat_completion_done_chunk_json(&req_id, &model_arg, &done);
                    chunk["created"] = json!(created);
                    chunk
                };
                if include_usage {
                    final_chunk["usage"] = openai_usage_json(&done);
                }
                final_chunk["timings"] = openai_timings_json(&done);
                let _ = tx
                    .send(Ok(
                        Event::default().data(serde_json::to_string(&final_chunk).unwrap())
                    ))
                    .await;
            }
            Ok(None) => {
                tracing::info!(request_id = %req_id, "stream client disconnected; dropping daemon");
                *state.loaded_model_path.lock().await = None;
                *state.loaded_model_cache_capable.lock().await = None;
                *state.loaded_model_max_seq.lock().await = None;
                drop(engine);
                return;
            }
            Err(e) => {
                *engine_guard = Some(engine);
                let _ = tx.send(Ok(sse_error(&e.to_string()))).await;
            }
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    let stream = async_stream::stream! {
        while let Some(item) = rx.recv().await {
            yield item;
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(10))
            .text("prefill"),
    )
}

fn sse_error(msg: &str) -> Event {
    Event::default().data(serde_json::to_string(&json!({"error": {"message": msg}})).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_prompt::Role;

    #[test]
    fn chat_messages_forward_as_structured_daemon_messages() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some(Value::String("be brief".to_string())),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("first".to_string())),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(Value::String("ok".to_string())),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("second".to_string())),
                ..Default::default()
            },
        ];

        let req = generate_request_from_chat(
            "req".to_string(),
            &messages,
            GenerationSamplingPolicy {
                temperature: 0.3,
                max_tokens: 16,
                top_p: Some(0.8),
                repeat_penalty: Some(1.0),
            },
            Some("worker-a".to_string()),
            Some(json!([{"type":"function"}])),
            Some("system override".to_string()),
            Some(vec!["END".to_string()]),
            Some("AAAA".to_string()),
            RequestGenerationControls {
                presence_penalty: Some(0.1),
                frequency_penalty: Some(0.2),
                reasoning_effort: Some("low".to_string()),
                thinking_mode: Some("thinking".to_string()),
                assistant_prefix: Some("open_think".to_string()),
                max_think_tokens: Some(256),
            },
        );
        let v = serde_json::to_value(&req).expect("serialize generate request");

        assert_eq!(v["prompt"], "second");
        assert!(!v["prompt"].as_str().unwrap().contains("<|im_start|>"));
        assert_eq!(v["messages"][0]["role"], "system");
        assert_eq!(v["messages"][0]["content"], "be brief");
        assert_eq!(v["messages"][1]["role"], "user");
        assert_eq!(v["messages"][3]["content"], "second");
        assert_eq!(v["worker_key_id"], "worker-a");
        assert_eq!(v["tools"][0]["type"], "function");
        assert_eq!(v["system"], "system override");
        assert_eq!(v["stop"], json!(["END"]));
        assert_eq!(v["image_base64"], "AAAA");
        assert_eq!(v["presence_penalty"], 0.1);
        assert_eq!(v["frequency_penalty"], 0.2);
        assert_eq!(v["reasoning_effort"], "low");
        assert_eq!(v["thinking_mode"], "thinking");
        assert_eq!(v["assistant_prefix"], "open_think");
        assert_eq!(v["max_think_tokens"], 256);
    }

    #[test]
    fn visible_thinking_is_stripped_from_non_stream_content() {
        assert_eq!(
            strip_visible_thinking(
                "<think>\nwork\n</think>\n\nanswer<|im_end|>".to_string(),
                false,
                false,
            ),
            "answer"
        );
        assert_eq!(
            strip_visible_thinking("\nwork\n</think>\n\nanswer".to_string(), false, true),
            "answer"
        );
        assert_eq!(
            strip_visible_thinking("plain answer".to_string(), false, true),
            "plain answer"
        );
        assert_eq!(
            strip_visible_thinking("<think>x</think> answer".to_string(), true, false),
            "<think>x</think> answer"
        );
    }

    #[test]
    fn think_stream_filter_splits_reasoning_from_content() {
        let mut filter = ThinkStreamFilter::default();
        let deltas = filter.observe("<think>\nwhy</think>\n\nanswer", false);

        assert!(matches!(&deltas[0], AssistantDelta::Reasoning(s) if s == "\nwhy"));
        assert!(matches!(&deltas[1], AssistantDelta::Content(s) if s == "answer"));
    }

    #[test]
    fn stream_options_include_usage_defaults_false() {
        let body: ChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .expect("chat request");

        assert!(include_stream_usage(&body));

        let body_without: ChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }))
        .expect("chat request");
        assert!(!include_stream_usage(&body_without));
    }

    #[test]
    fn last_user_prompt_is_compatibility_fallback_only() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("first".to_string())),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(Value::String("answer".to_string())),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(json!({"type":"text","text":"second"})),
                ..Default::default()
            },
        ];

        let req = generate_request_from_chat(
            "req".to_string(),
            &messages,
            GenerationSamplingPolicy::greedy(8),
            None,
            None,
            None,
            None,
            None,
            RequestGenerationControls::default(),
        );

        let prompt_value: Value =
            serde_json::from_str(&req.prompt).expect("structured prompt json");
        assert_eq!(prompt_value, json!({"type":"text","text":"second"}));
        let daemon_messages = req.messages.unwrap();
        assert_eq!(daemon_messages.len(), 3);
        assert_eq!(daemon_messages[2].role, Role::User);
    }

    #[test]
    fn chat_message_aliases_and_tool_call_history_match_bun_mapping() {
        let messages = vec![
            ChatMessage {
                role: "developer".to_string(),
                content: Some(Value::String("follow policy".to_string())),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("lookup hipfire".to_string())),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(Value::Null),
                tool_calls: Some(vec![json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"q\":\"hipfire\"}"
                    }
                })]),
                ..Default::default()
            },
            ChatMessage {
                role: "toolResult".to_string(),
                content: Some(Value::String("fast local inference".to_string())),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
        ];

        let req = generate_request_from_chat(
            "req".to_string(),
            &messages,
            GenerationSamplingPolicy::greedy(8),
            None,
            None,
            None,
            None,
            None,
            RequestGenerationControls::default(),
        );

        let daemon_messages = req.messages.unwrap();
        assert_eq!(daemon_messages[0].role, Role::System);
        assert_eq!(daemon_messages[0].content, "follow policy");
        assert_eq!(daemon_messages[2].role, Role::Assistant);
        assert_eq!(daemon_messages[2].content, "");
        assert_eq!(daemon_messages[2].tool_calls.len(), 1);
        assert_eq!(daemon_messages[2].tool_calls[0].name, "lookup");
        assert_eq!(
            daemon_messages[2].tool_calls[0].arguments,
            json!({"q":"hipfire"})
        );
        assert_eq!(daemon_messages[3].role, Role::Tool);
        assert_eq!(daemon_messages[3].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn stop_sequences_match_bun_normalization() {
        let value = json!(["", "END", 7, "ignored"]);
        assert!(normalize_stop_sequences(Some(&value)).is_err());

        let long = "x".repeat(80);
        let value = json!(["", "END", long, "A", "B", "C"]);
        let stop = normalize_stop_sequences(Some(&value)).unwrap().unwrap();
        assert_eq!(stop.len(), 4);
        assert_eq!(stop[0], "END");
        assert_eq!(stop[1].len(), 64);

        let one = normalize_stop_sequences(Some(&json!("DONE")))
            .unwrap()
            .unwrap();
        assert_eq!(one, vec!["DONE"]);
    }

    #[test]
    fn image_url_parts_forward_single_last_user_data_uri() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!([
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ])),
            ..Default::default()
        }];

        assert_eq!(
            extract_request_image_base64(&messages).unwrap(),
            Some("AAAA".to_string())
        );
    }

    #[test]
    fn image_url_parts_reject_unsupported_shapes() {
        let remote = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!([
                {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
            ])),
            ..Default::default()
        }];
        assert!(extract_request_image_base64(&remote)
            .unwrap_err()
            .contains("remote image URLs"));

        let unsupported = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!([
                {"type": "image_url", "image_url": {"url": "data:image/webp;base64,AAAA"}}
            ])),
            ..Default::default()
        }];
        assert!(extract_request_image_base64(&unsupported)
            .unwrap_err()
            .contains("unsupported image format"));

        let earlier = vec![
            ChatMessage {
                role: "user".to_string(),
                content: Some(json!([
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}
                ])),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(json!("ok")),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(json!("next")),
                ..Default::default()
            },
        ];
        assert!(extract_request_image_base64(&earlier)
            .unwrap_err()
            .contains("earlier user turns"));
    }

    #[test]
    fn inline_tool_call_text_parses_to_openai_tool_calls() {
        let (content, tool_calls) = parse_inline_tool_calls(
            "Before\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"hipfire\"}}</tool_call>",
            "req",
        );

        assert_eq!(content, "Before");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["type"], "function");
        assert_eq!(tool_calls[0]["function"]["name"], "lookup");
        assert_eq!(
            tool_calls[0]["function"]["arguments"],
            serde_json::to_string(&json!({"q": "hipfire"})).unwrap()
        );
    }

    #[test]
    fn native_xml_tool_call_text_parses_to_openai_tool_calls() {
        let (_, tool_calls) = parse_inline_tool_calls(
            "<tool_call><function=write><parameter=path>README.md</parameter><parameter=overwrite>true</parameter></function></tool_call>",
            "req",
        );

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["function"]["name"], "write");
        let args: Value =
            serde_json::from_str(tool_calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"path": "README.md", "overwrite": true}));
    }

    #[test]
    fn malformed_tool_call_shapes_match_bun_recovery() {
        let (_, flat_calls) = parse_inline_tool_calls(
            "<tool_call>{\"name\":\"write\",\"path\":\"README.md\",\"content\":\"hi\"}</tool_call>",
            "req",
        );
        let flat_args: Value =
            serde_json::from_str(flat_calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(flat_args, json!({"path": "README.md", "content": "hi"}));

        let (_, xml_calls) = parse_inline_tool_calls(
            "<tool_call><plain>write</param> {\"path\":\"README.md\"}</tool_call>",
            "req",
        );
        assert_eq!(xml_calls[0]["function"]["name"], "write");

        let (_, fallback_calls) = parse_inline_tool_calls(
            "<tool_call><|im_start|>name\": \"lookup\", \"arguments\": {\"q\":\"hipfire\"}}</tool_call>",
            "req",
        );
        assert_eq!(fallback_calls[0]["function"]["name"], "lookup");
        let fallback_args: Value =
            serde_json::from_str(fallback_calls[0]["function"]["arguments"].as_str().unwrap())
                .unwrap();
        assert_eq!(fallback_args, json!({"q": "hipfire"}));
    }

    #[test]
    fn chat_response_includes_tool_calls_finish_reason() {
        let done = hipfire_generate::DoneEvent {
            id: "req".to_string(),
            tokens: 5,
            tok_s: None,
            prefill_tokens: Some(3),
            prefill_ms: None,
            prefill_tok_s: None,
            decode_tok_s: None,
            ttft_ms: None,
            finish_reason: Some("stop".to_string()),
            response_id: None,
            extra: Default::default(),
        };
        let body = openai_chat_completion_response_with_tool_calls_json(
            "req",
            12345,
            "qwen",
            "",
            &[json!({
                "id": "call_req_0",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"}
            })],
            &done,
            512,
        );

        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(body["created"], 12345);
        assert_eq!(body["choices"][0]["message"]["content"], Value::Null);
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "lookup"
        );
    }

    #[test]
    fn nonstream_response_flags_truncated_tool_call() {
        let done = hipfire_generate::DoneEvent {
            id: "req".to_string(),
            tokens: 64,
            tok_s: None,
            prefill_tokens: Some(3),
            prefill_ms: None,
            prefill_tok_s: None,
            decode_tok_s: None,
            ttft_ms: None,
            finish_reason: Some("stop".to_string()),
            response_id: None,
            extra: Default::default(),
        };

        let body = openai_chat_completion_response_with_tool_calls_json(
            "req",
            12345,
            "qwen",
            "<tool_call>{\"name\":\"write\",\"arguments\":{\"path\":\"README.md\"",
            &[],
            &done,
            64,
        );

        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["truncation"]["reason"], "max_tokens_in_tool_call");
        assert_eq!(body["truncation"]["max_tokens_used"], 64);
        assert_eq!(body["truncation"]["suggested_max_tokens"], 4096);
    }

    #[test]
    fn chat_usage_includes_cache_details_from_daemon_extras() {
        let done = hipfire_generate::DoneEvent {
            id: "req".to_string(),
            tokens: 5,
            tok_s: None,
            prefill_tokens: Some(3),
            prefill_ms: None,
            prefill_tok_s: None,
            decode_tok_s: None,
            ttft_ms: None,
            finish_reason: Some("stop".to_string()),
            response_id: None,
            extra: std::collections::HashMap::from([
                ("prompt_tokens".to_string(), json!(11)),
                ("cached_tokens".to_string(), json!(8)),
            ]),
        };

        let usage = openai_usage_json(&done);

        assert_eq!(usage["prompt_tokens"], 11);
        assert_eq!(usage["completion_tokens"], 5);
        assert_eq!(usage["total_tokens"], 16);
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 8);
        assert_eq!(usage["prompt_tokens_details"]["cache_write_tokens"], 3);
        assert_eq!(usage["cache_read_input_tokens"], 8);
        assert_eq!(usage["cache_creation_input_tokens"], 3);
    }

    #[test]
    fn nonstream_chat_usage_matches_bun_cache_write_fallback() {
        let done = hipfire_generate::DoneEvent {
            id: "req".to_string(),
            tokens: 2,
            tok_s: None,
            prefill_tokens: Some(99),
            prefill_ms: None,
            prefill_tok_s: None,
            decode_tok_s: None,
            ttft_ms: None,
            finish_reason: Some("stop".to_string()),
            response_id: None,
            extra: std::collections::HashMap::from([
                ("prompt_tokens".to_string(), json!(12)),
                ("cached_tokens".to_string(), json!(7)),
            ]),
        };

        let body = openai_chat_completion_response_with_tool_calls_json(
            "req",
            12345,
            "qwen",
            "hi",
            &[],
            &done,
            512,
        );

        assert_eq!(body["created"], 12345);
        assert_eq!(body["usage"]["prompt_tokens"], 12);
        assert_eq!(body["usage"]["completion_tokens"], 2);
        assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 7);
        assert_eq!(
            body["usage"]["prompt_tokens_details"]["cache_write_tokens"],
            5
        );
        assert_eq!(body["usage"]["cache_creation_input_tokens"], 5);
    }

    #[test]
    fn streaming_final_timings_include_daemon_metrics_and_spec_extras() {
        let done = hipfire_generate::DoneEvent {
            id: "req".to_string(),
            tokens: 13,
            tok_s: Some(44.5),
            prefill_tokens: Some(21),
            prefill_ms: Some(7.5),
            prefill_tok_s: Some(2800.0),
            decode_tok_s: Some(40.0),
            ttft_ms: Some(9.0),
            finish_reason: Some("stop".to_string()),
            response_id: None,
            extra: std::collections::HashMap::from([
                ("tau".to_string(), json!(2.5)),
                ("cycles".to_string(), json!(4)),
                ("dflash".to_string(), json!(true)),
            ]),
        };

        let timings = openai_timings_json(&done);

        assert_eq!(timings["tokens"], 13);
        assert_eq!(timings["tok_s"], 44.5);
        assert_eq!(timings["prefill_tokens"], 21);
        assert_eq!(timings["prefill_ms"], 7.5);
        assert_eq!(timings["prefill_tok_s"], 2800.0);
        assert_eq!(timings["decode_tok_s"], 40.0);
        assert_eq!(timings["ttft_ms"], 9.0);
        assert_eq!(timings["tau"], 2.5);
        assert_eq!(timings["cycles"], 4);
        assert_eq!(timings["dflash"], true);
    }

    #[test]
    fn streaming_tool_call_chunk_matches_openai_delta_shape() {
        let chunk = openai_chat_completion_tool_call_chunk_json(
            "req",
            12345,
            "qwen",
            2,
            &json!({
                "id": "call_req_2",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"}
            }),
        );

        assert_eq!(chunk["created"], 12345);
        assert_eq!(chunk["choices"][0]["delta"]["tool_calls"][0]["index"], 2);
        assert_eq!(
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "lookup"
        );
        assert_eq!(chunk["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn tool_presence_treats_empty_array_as_absent() {
        assert!(!tools_present(None));
        assert!(!tools_present(Some(&Value::Null)));
        assert!(!tools_present(Some(&json!([]))));
        assert!(tools_present(Some(&json!([{"type": "function"}]))));
        assert!(tools_present(Some(&json!({"type": "function"}))));
    }

    #[tokio::test]
    async fn blocking_chat_preflight_maps_early_errors_to_http_statuses() {
        let state = crate::state::AppState::new(HipfireConfig::default());

        let bad_stop = ChatRequest {
            model: Some("qwen3.5-0.8b-mq4".to_string()),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(json!("hi")),
                ..Default::default()
            }],
            stop: Some(json!(["ok", 3])),
            ..Default::default()
        };
        let (status, body) = blocking_chat_preflight_error(&state, &bad_stop)
            .await
            .expect("bad stop preflight");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error");

        let bad_image = ChatRequest {
            model: Some("qwen3.5-0.8b-mq4".to_string()),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(json!([
                    {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
                ])),
                ..Default::default()
            }],
            ..Default::default()
        };
        let (status, body) = blocking_chat_preflight_error(&state, &bad_image)
            .await
            .expect("bad image preflight");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("remote image URLs"));

        let missing_model = ChatRequest {
            model: Some("__definitely_missing_hipfire_model__".to_string()),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(json!("hi")),
                ..Default::default()
            }],
            ..Default::default()
        };
        let (status, body) = blocking_chat_preflight_error(&state, &missing_model)
            .await
            .expect("missing model preflight");
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("model not found"));
    }

    #[test]
    fn load_params_from_config_preserves_explicit_dflash_off() {
        let cfg = HipfireConfig {
            max_seq: 8192,
            kv_cache: "auto".to_string(),
            flash_mode: "auto".to_string(),
            dflash_mode: "off".to_string(),
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            cask_sidecar: Some("/models/qwen3.5-27b.triattn.hfq".to_string()),
            ..Default::default()
        };

        let params = load_params_from_config(&cfg);

        assert_eq!(params.max_seq, 8192);
        assert_eq!(params.kv_cache, None);
        assert_eq!(params.flash_mode, None);
        assert_eq!(params.dflash_mode.as_deref(), Some("off"));
        assert_eq!(params.mtp_mode.as_deref(), Some("auto"));
        assert_eq!(params.mtp_k, Some(3));
        assert_eq!(
            params.cask_sidecar.as_deref(),
            Some("/models/qwen3.5-27b.triattn.hfq")
        );
    }

    #[test]
    fn load_params_from_config_omits_auto_and_empty_sidecar() {
        let cfg = HipfireConfig {
            max_seq: 4096,
            kv_cache: "asym3".to_string(),
            flash_mode: "auto".to_string(),
            dflash_mode: "auto".to_string(),
            cask_sidecar: Some(String::new()),
            ..Default::default()
        };

        let params = load_params_from_config(&cfg);

        assert_eq!(params.max_seq, 4096);
        assert_eq!(params.kv_cache.as_deref(), Some("asym3"));
        assert_eq!(params.flash_mode, None);
        assert_eq!(params.dflash_mode, None);
        assert_eq!(params.cask_sidecar, None);
    }

    #[test]
    fn load_params_resolve_per_model_overrides_like_bun() {
        let mut cfg = HipfireConfig {
            max_seq: 4096,
            kv_cache: "auto".to_string(),
            flash_mode: "auto".to_string(),
            dflash_mode: "off".to_string(),
            ..Default::default()
        };
        cfg.model_overrides.insert(
            "qwen3.5-0.8b-mq4".to_string(),
            json!({
                "max_seq": 8192,
                "kv_cache": "asym4",
                "flash_mode": "on",
                "dflash_mode": "on",
                "mtp_k": 5
            }),
        );

        let params = load_params_for_model_config(&cfg, "qwen3.5-0.8b-mq4", None);

        assert_eq!(params.max_seq, 8192);
        assert_eq!(params.kv_cache.as_deref(), Some("asym4"));
        assert_eq!(params.flash_mode.as_deref(), Some("on"));
        assert_eq!(params.dflash_mode.as_deref(), Some("on"));
        assert_eq!(params.mtp_k, Some(5));
    }

    #[test]
    fn dflash_ngram_block_matches_bun_auto_small_dense_policy() {
        assert!(resolve_dflash_ngram_block(
            &json!("auto"),
            "qwen3.5-0.8b-mq4"
        ));
        assert!(resolve_dflash_ngram_block(&json!("auto"), "qwen3.5:4b"));
        assert!(!resolve_dflash_ngram_block(
            &json!("auto"),
            "qwen3.5-9b-mq4"
        ));
        assert!(!resolve_dflash_ngram_block(
            &json!("auto"),
            "qwen3.5-35b-a3b-mq4"
        ));
        assert!(resolve_dflash_ngram_block(&json!(true), "qwen3.5-27b-mq4"));
        assert!(!resolve_dflash_ngram_block(
            &json!(false),
            "qwen3.5-0.8b-mq4"
        ));
    }

    #[test]
    fn daemon_spawn_env_uses_per_model_dflash_ngram_override() {
        let mut cfg = HipfireConfig::default();
        cfg.model_overrides.insert(
            "qwen3.5-27b-mq4".to_string(),
            json!({ "prompt_normalize": false, "dflash_ngram_block": true }),
        );

        let resolved = cfg.resolve_for_model("qwen3.5-27b-mq4");
        let env = DaemonSpawnEnv::from_resolved_config(&resolved, "qwen3.5-27b-mq4");

        assert!(!env.prompt_normalize);
        assert!(env.dflash_ngram_block);
    }

    #[test]
    fn load_params_bump_max_seq_to_cover_generation_budget_like_bun() {
        let cfg = HipfireConfig {
            max_seq: 1024,
            max_tokens: 4096,
            ..Default::default()
        };

        let params = load_params_for_model_config(&cfg, "qwen3.5-0.8b-mq4", None);

        assert_eq!(params.max_seq, 5120);
    }

    #[test]
    fn load_params_attach_discovered_dflash_draft_like_bun() {
        let root = std::env::temp_dir().join(format!(
            "hipfire-server-dflash-draft-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("qwen3.5-27b-mq4.hfq");
        let draft = root.join("qwen3.5-27b-mq4.dflash.hfq");
        std::fs::write(&target, "target").unwrap();
        std::fs::write(&draft, "draft").unwrap();

        let cfg = HipfireConfig {
            dflash_mode: "auto".to_string(),
            ..Default::default()
        };
        let params = load_params_for_model_config(&cfg, "qwen3.5-27b-mq4", Some(&target));

        assert_eq!(params.draft.as_deref(), Some(draft.to_str().unwrap()));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_params_drop_missing_explicit_cask_sidecar_like_bun() {
        let cfg = HipfireConfig {
            cask_sidecar: Some("/definitely/missing/model.triattn.hfq".to_string()),
            cask: true,
            ..Default::default()
        };

        let params = load_params_for_model_config(&cfg, "qwen3.5-27b-mq4", None);

        assert_eq!(params.cask_sidecar, None);
        assert_eq!(params.cask, None);
    }

    #[test]
    fn load_params_auto_attach_triattn_sidecar_like_bun() {
        let root = std::env::temp_dir().join(format!(
            "hipfire-server-triattn-sidecar-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("qwen3.5-27b-mq4.hfq");
        let sidecar = root.join("qwen3.5-27b-mq4.hfq.triattn.hfq");
        std::fs::write(&target, "target").unwrap();
        std::fs::write(&sidecar, "sidecar").unwrap();
        let cfg = HipfireConfig {
            cask_auto_attach: true,
            cask_budget: 2048,
            ..Default::default()
        };

        let params = load_params_for_model_config(&cfg, "qwen3.5-27b-mq4", Some(&target));

        assert_eq!(
            params.cask_sidecar.as_deref(),
            Some(sidecar.to_str().unwrap())
        );
        assert_eq!(params.cask_budget, Some(2048));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn request_sizing_uses_per_model_generation_defaults() {
        let mut cfg = HipfireConfig {
            max_seq: 4096,
            max_tokens: 512,
            ..Default::default()
        };
        cfg.model_overrides.insert(
            "big".to_string(),
            json!({
                "max_seq": 16384,
                "max_tokens": 8192
            }),
        );
        let resolved = cfg.resolve_for_model("big");
        let request_max_tokens = effective_request_max_tokens(resolved.max_tokens, None);
        let required_max_seq = required_load_max_seq(resolved.max_seq, request_max_tokens, false);

        assert_eq!(request_max_tokens, 8192);
        assert_eq!(required_max_seq, 16384);
    }

    #[test]
    fn request_generation_controls_match_bun_reasoning_mapping() {
        let mut cfg = HipfireConfig::default();
        cfg.thinking = "on".to_string();

        let controls = request_generation_controls(
            &cfg,
            Some(&json!({"enable_thinking": false})),
            None,
            None,
            Some(-1.0),
            Some(0.4),
        );
        assert_eq!(controls.assistant_prefix.as_deref(), Some("closed_think"));
        assert_eq!(controls.thinking_mode.as_deref(), Some("chat"));
        assert_eq!(controls.max_think_tokens, Some(1));
        assert_eq!(controls.presence_penalty, Some(0.0));
        assert_eq!(controls.frequency_penalty, Some(0.4));

        let controls = request_generation_controls(
            &cfg,
            None,
            Some("high"),
            Some(&json!({"effort": "none"})),
            None,
            None,
        );
        assert_eq!(controls.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(controls.assistant_prefix.as_deref(), Some("open_think"));
        assert_eq!(controls.thinking_mode.as_deref(), Some("thinking"));
        assert_eq!(controls.max_think_tokens, Some(4096));

        let controls = request_generation_controls(
            &cfg,
            None,
            None,
            Some(&json!({"effort": "xhigh"})),
            None,
            None,
        );
        assert_eq!(controls.reasoning_effort.as_deref(), Some("xhigh"));
        assert_eq!(controls.thinking_mode.as_deref(), Some("max"));
        assert_eq!(controls.max_think_tokens, None);
    }

    #[test]
    fn loaded_response_cache_capability_matches_bun_arch_policy() {
        let mut loaded = hipfire_model::ModelLoadedResponse {
            worker_key_id: "worker-a".to_string(),
            arch: Some("qwen3_5".to_string()),
            cache_capable: None,
            dim: None,
            layers: None,
            vocab: None,
            model_worker: None,
            response_id: None,
        };
        assert!(loaded_response_cache_capable(&loaded));

        loaded.arch = Some("deepseek4".to_string());
        assert!(loaded_response_cache_capable(&loaded));

        loaded.arch = Some("qwen2".to_string());
        assert!(!loaded_response_cache_capable(&loaded));

        loaded.cache_capable = Some(true);
        assert!(loaded_response_cache_capable(&loaded));

        loaded.cache_capable = Some(false);
        loaded.arch = Some("qwen3_5".to_string());
        assert!(!loaded_response_cache_capable(&loaded));
    }

    #[test]
    fn request_max_tokens_and_load_max_seq_match_bun_sizing_policy() {
        assert_eq!(effective_request_max_tokens(512, None), 512);
        assert_eq!(effective_request_max_tokens(512, Some(0)), 512);
        assert_eq!(effective_request_max_tokens(512, Some(4096)), 4096);
        assert_eq!(
            effective_request_max_tokens(512, Some(MAX_REQUEST_TOKENS + 1)),
            512
        );

        assert_eq!(required_load_max_seq(4096, 512, false), 4096);
        assert_eq!(required_load_max_seq(4096, 8192, false), 9216);
        assert_eq!(required_load_max_seq(4096, 8192, true), 10240);
        assert_eq!(
            required_load_max_seq(4096, MAX_LOAD_MAX_SEQ, true),
            MAX_LOAD_MAX_SEQ
        );
    }
}
