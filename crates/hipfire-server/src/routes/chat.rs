use std::convert::Infallible;

use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json, Response,
    },
};
use hipfire_prompt::{Message as DaemonMessage, Role};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::config::HipfireConfig;
use crate::daemon::engine::{find_daemon_bin, DaemonEngine};
use crate::daemon::protocol::{GenerateRequest, GenerationSamplingPolicy, LoadParams};
use crate::model::discovery::find_model;
use crate::state::SharedState;

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub tools: Option<Value>,
    pub system: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<Value>,
}

pub async fn post_chat_completions(
    State(state): State<SharedState>,
    Json(body): Json<ChatRequest>,
) -> Response {
    if body.stream {
        stream_chat(state, body).await.into_response()
    } else {
        blocking_chat(state, body).await.into_response()
    }
}

fn message_content_to_text(content: &Option<Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

fn chat_role_to_daemon(role: &str) -> Option<Role> {
    match role {
        "system" => Some(Role::System),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        "tool" => Some(Role::Tool),
        _ => None,
    }
}

fn messages_to_daemon(messages: &[ChatMessage]) -> Vec<DaemonMessage> {
    messages
        .iter()
        .filter_map(|m| {
            Some(DaemonMessage {
                role: chat_role_to_daemon(&m.role)?,
                content: message_content_to_text(&m.content),
                tool_calls: Vec::new(),
                tool_call_id: None,
            })
        })
        .collect()
}

fn last_user_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| message_content_to_text(&m.content))
        .unwrap_or_default()
}

fn load_params_from_config(cfg: &HipfireConfig) -> LoadParams {
    LoadParams {
        max_seq: cfg.max_seq,
        kv_cache: Some(cfg.kv_cache.clone()).filter(|s| s != "auto"),
        flash_mode: Some(cfg.flash_mode.clone()).filter(|s| s != "auto"),
        dflash_mode: Some(cfg.dflash_mode.clone()).filter(|s| s != "auto"),
        cask_sidecar: cfg
            .cask_sidecar
            .clone()
            .filter(|sidecar| !sidecar.is_empty()),
        ..Default::default()
    }
}

async fn ensure_model_loaded(state: &SharedState, model_arg: &str) -> Result<(), String> {
    let model_path =
        find_model(model_arg).ok_or_else(|| format!("model not found: {model_arg}"))?;
    let model_str = model_path.to_string_lossy().into_owned();

    let mut engine_guard = state.engine.lock().await;
    let mut loaded_guard = state.loaded_model_path.lock().await;

    if loaded_guard.as_deref() == Some(&model_str) {
        if let Some(eng) = engine_guard.as_mut() {
            if eng.ping().await.is_ok() {
                return Ok(());
            }
        }
    }

    let bin = find_daemon_bin().ok_or_else(|| {
        "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`"
            .to_string()
    })?;

    let mut engine = DaemonEngine::spawn(&bin).await.map_err(|e| e.to_string())?;

    let params = {
        let cfg = state.config.lock().await;
        load_params_from_config(&cfg)
    };

    engine
        .load(&model_str, params)
        .await
        .map_err(|e| e.to_string())?;

    *loaded_guard = Some(model_str);
    *engine_guard = Some(engine);
    Ok(())
}

async fn blocking_chat(state: SharedState, body: ChatRequest) -> impl IntoResponse {
    let req_id = Uuid::new_v4().to_string();

    let model_arg = {
        let cfg = state.config.lock().await;
        body.model.clone().or(cfg.default_model.clone())
    };

    let Some(model_arg) = model_arg else {
        return Json(
            json!({"error": {"message": "no model specified", "type": "invalid_request_error"}}),
        )
        .into_response();
    };

    if let Err(e) = ensure_model_loaded(&state, &model_arg).await {
        return Json(json!({"error": {"message": e, "type": "server_error"}})).into_response();
    }

    let gen_req = {
        let cfg = state.config.lock().await;
        let worker_key_id = state
            .engine
            .lock()
            .await
            .as_ref()
            .and_then(|e| e.worker_key_id.clone());
        GenerateRequest {
            id: req_id.clone(),
            prompt: last_user_prompt(&body.messages),
            messages: Some(messages_to_daemon(&body.messages)),
            sampling: GenerationSamplingPolicy {
                temperature: body.temperature.unwrap_or(cfg.temperature),
                max_tokens: body.max_tokens.unwrap_or(cfg.max_tokens),
                top_p: Some(body.top_p.unwrap_or(cfg.top_p)),
                repeat_penalty: Some(cfg.repeat_penalty),
            },
            worker_key_id,
            tools: body.tools,
            system: body.system,
            thinking: None,
            max_think_tokens: None,
            request_id: None,
        }
    };

    let mut engine_guard = state.engine.lock().await;
    let engine = match engine_guard.as_mut() {
        Some(e) => e,
        None => {
            return Json(
                json!({"error": {"message": "daemon not running", "type": "server_error"}}),
            )
            .into_response()
        }
    };

    match engine.generate(gen_req).await {
        Ok((text, done)) => {
            let finish_reason = done.finish_reason.as_deref().unwrap_or("stop");
            let prompt_tokens = done.prefill_tokens.unwrap_or(0);
            let completion_tokens = done.tokens;
            Json(json!({
                "id": format!("chatcmpl-{req_id}"),
                "object": "chat.completion",
                "model": model_arg,
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": text },
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            }))
            .into_response()
        }
        Err(e) => Json(json!({"error": {"message": e.to_string(), "type": "server_error"}}))
            .into_response(),
    }
}

async fn stream_chat(state: SharedState, body: ChatRequest) -> impl IntoResponse {
    let (tx, mut rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        let req_id = Uuid::new_v4().to_string();

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

        if let Err(e) = ensure_model_loaded(&state, &model_arg).await {
            let _ = tx.send(Ok(sse_error(&e))).await;
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
            return;
        }

        let gen_req = {
            let cfg = state.config.lock().await;
            let worker_key_id = state
                .engine
                .lock()
                .await
                .as_ref()
                .and_then(|e| e.worker_key_id.clone());
            GenerateRequest {
                id: req_id.clone(),
                prompt: last_user_prompt(&body.messages),
                messages: Some(messages_to_daemon(&body.messages)),
                sampling: GenerationSamplingPolicy {
                    temperature: body.temperature.unwrap_or(cfg.temperature),
                    max_tokens: body.max_tokens.unwrap_or(cfg.max_tokens),
                    top_p: Some(body.top_p.unwrap_or(cfg.top_p)),
                    repeat_penalty: Some(cfg.repeat_penalty),
                },
                worker_key_id,
                tools: body.tools,
                system: body.system,
                thinking: None,
                max_think_tokens: None,
                request_id: None,
            }
        };

        let req_id_cb = req_id.clone();
        let model_cb = model_arg.clone();
        let tx_cb = tx.clone();

        let mut engine_guard = state.engine.lock().await;
        let engine = match engine_guard.as_mut() {
            Some(e) => e,
            None => {
                let _ = tx.send(Ok(sse_error("daemon not running"))).await;
                let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                return;
            }
        };

        let result = engine
            .generate_streaming(gen_req, move |token| {
                let chunk = json!({
                    "id": format!("chatcmpl-{req_id_cb}"),
                    "object": "chat.completion.chunk",
                    "model": model_cb,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": token}, "finish_reason": null}]
                });
                let _ = tx_cb.try_send(Ok(Event::default().data(serde_json::to_string(&chunk).unwrap())));
            })
            .await;

        if let Ok(done) = result {
            let finish_reason = done.finish_reason.as_deref().unwrap_or("stop");
            let final_chunk = json!({
                "id": format!("chatcmpl-{req_id}"),
                "object": "chat.completion.chunk",
                "model": model_arg,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
            });
            let _ = tx
                .send(Ok(
                    Event::default().data(serde_json::to_string(&final_chunk).unwrap())
                ))
                .await;
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    let stream = async_stream::stream! {
        while let Some(item) = rx.recv().await {
            yield item;
        }
    };

    Sse::new(stream)
}

fn sse_error(msg: &str) -> Event {
    Event::default().data(serde_json::to_string(&json!({"error": {"message": msg}})).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_messages_forward_as_structured_daemon_messages() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some(Value::String("be brief".to_string())),
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("first".to_string())),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(Value::String("ok".to_string())),
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("second".to_string())),
            },
        ];

        let req = GenerateRequest {
            id: "req".to_string(),
            prompt: last_user_prompt(&messages),
            messages: Some(messages_to_daemon(&messages)),
            sampling: GenerationSamplingPolicy {
                temperature: 0.3,
                max_tokens: 16,
                top_p: Some(0.8),
                repeat_penalty: Some(1.0),
            },
            worker_key_id: None,
            tools: None,
            system: None,
            thinking: None,
            max_think_tokens: None,
            request_id: None,
        };
        let v = serde_json::to_value(&req).expect("serialize generate request");

        assert_eq!(v["prompt"], "second");
        assert!(!v["prompt"].as_str().unwrap().contains("<|im_start|>"));
        assert_eq!(v["messages"][0]["role"], "system");
        assert_eq!(v["messages"][0]["content"], "be brief");
        assert_eq!(v["messages"][1]["role"], "user");
        assert_eq!(v["messages"][3]["content"], "second");
    }

    #[test]
    fn last_user_prompt_is_compatibility_fallback_only() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("first".to_string())),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(Value::String("answer".to_string())),
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(json!({"type":"text","text":"second"})),
            },
        ];

        let prompt_value: Value =
            serde_json::from_str(&last_user_prompt(&messages)).expect("structured prompt json");
        assert_eq!(prompt_value, json!({"type":"text","text":"second"}));
        let daemon_messages = messages_to_daemon(&messages);
        assert_eq!(daemon_messages.len(), 3);
        assert_eq!(daemon_messages[2].role, Role::User);
    }

    #[test]
    fn load_params_from_config_preserves_explicit_dflash_off() {
        let cfg = HipfireConfig {
            max_seq: 8192,
            kv_cache: "auto".to_string(),
            flash_mode: "auto".to_string(),
            dflash_mode: "off".to_string(),
            cask_sidecar: Some("/models/qwen3.5-27b.triattn.hfq".to_string()),
            ..Default::default()
        };

        let params = load_params_from_config(&cfg);

        assert_eq!(params.max_seq, 8192);
        assert_eq!(params.kv_cache, None);
        assert_eq!(params.flash_mode, None);
        assert_eq!(params.dflash_mode.as_deref(), Some("off"));
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
}
