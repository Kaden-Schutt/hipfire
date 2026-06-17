use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
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
use hipfire_daemon_adapter::{find_daemon_bin_or_error, DaemonEngine};
use hipfire_generate::{
    openai_chat_completion_done_chunk_json, openai_chat_completion_response_json,
    openai_chat_completion_token_chunk_json, GenerateTextRequest, GenerationSamplingPolicy,
};
use hipfire_model::{ModelLoadParams, ModelWorkerKey};
use hipfire_scheduler::{
    create_request_session_draft, server_prefill_batch_enabled, CreateRequestSessionInput,
    NextBatchInput, SchedulerPolicyEnv,
};

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub priority: Option<i64>,
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

fn load_params_from_config(cfg: &HipfireConfig) -> ModelLoadParams {
    ModelLoadParams::from_hipfire_config(cfg)
}

#[derive(Clone, Debug)]
struct LoadedModelContext {
    model_path: String,
    worker_key_id: Option<String>,
}

fn generate_request_from_chat(
    id: String,
    messages: &[ChatMessage],
    sampling: GenerationSamplingPolicy,
    worker_key_id: Option<String>,
    tools: Option<Value>,
    system: Option<String>,
) -> GenerateTextRequest {
    GenerateTextRequest::from_openai_chat_messages(
        id,
        messages
            .iter()
            .map(|message| (message.role.as_str(), message.content.as_ref())),
        sampling,
    )
    .with_worker_key_id(worker_key_id)
    .with_tools(tools)
    .with_system(system)
}

async fn ensure_model_loaded(
    state: &SharedState,
    model_arg: &str,
) -> Result<LoadedModelContext, String> {
    let model_path =
        find_model(model_arg).ok_or_else(|| format!("model not found: {model_arg}"))?;
    let model_str = model_path.to_string_lossy().into_owned();

    let mut engine_guard = state.engine.lock().await;
    let mut loaded_guard = state.loaded_model_path.lock().await;

    if loaded_guard.as_deref() == Some(&model_str) {
        if let Some(eng) = engine_guard.as_mut() {
            if eng.ping().await.is_ok() {
                return Ok(LoadedModelContext {
                    model_path: model_str,
                    worker_key_id: eng.worker_key_id.clone(),
                });
            }
        }
    }

    let bin = find_daemon_bin_or_error().map_err(|e| e.to_string())?;

    let mut engine = DaemonEngine::spawn(&bin).await.map_err(|e| e.to_string())?;

    let params = {
        let cfg = state.config.lock().await;
        load_params_from_config(&cfg)
    };

    let loaded = engine
        .load(&model_str, params)
        .await
        .map_err(|e| e.to_string())?;

    let worker_key_id = Some(loaded.worker_key_id);
    *loaded_guard = Some(model_str);
    *engine_guard = Some(engine);
    Ok(LoadedModelContext {
        model_path: loaded_guard
            .as_ref()
            .expect("loaded model path set")
            .clone(),
        worker_key_id,
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
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

async fn wait_for_prefill_scheduler_turn(
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

    let loaded = match ensure_model_loaded(&state, &model_arg).await {
        Ok(loaded) => loaded,
        Err(e) => {
            return Json(json!({"error": {"message": e, "type": "server_error"}})).into_response()
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
        return Json(json!({"error": {"message": e, "type": "server_error"}})).into_response();
    }

    let gen_req = {
        let cfg = state.config.lock().await;
        generate_request_from_chat(
            req_id.clone(),
            &body.messages,
            GenerationSamplingPolicy::from_defaults(
                cfg.temperature,
                cfg.top_p,
                cfg.repeat_penalty,
                cfg.max_tokens,
                body.temperature,
                body.top_p,
                body.max_tokens,
            ),
            loaded.worker_key_id,
            body.tools,
            body.system,
        )
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
        Ok((text, done)) => Json(openai_chat_completion_response_json(
            &req_id, &model_arg, &text, &done,
        ))
        .into_response(),
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

        let loaded = match ensure_model_loaded(&state, &model_arg).await {
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
            generate_request_from_chat(
                req_id.clone(),
                &body.messages,
                GenerationSamplingPolicy::from_defaults(
                    cfg.temperature,
                    cfg.top_p,
                    cfg.repeat_penalty,
                    cfg.max_tokens,
                    body.temperature,
                    body.top_p,
                    body.max_tokens,
                ),
                loaded.worker_key_id,
                body.tools,
                body.system,
            )
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
                let chunk = openai_chat_completion_token_chunk_json(&req_id_cb, &model_cb, &token);
                let _ = tx_cb.try_send(Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                ));
            })
            .await;

        if let Ok(done) = result {
            let final_chunk = openai_chat_completion_done_chunk_json(&req_id, &model_arg, &done);
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
    use hipfire_prompt::Role;

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

        let req = generate_request_from_chat(
            "req".to_string(),
            &messages,
            GenerationSamplingPolicy::greedy(8),
            None,
            None,
            None,
        );

        let prompt_value: Value =
            serde_json::from_str(&req.prompt).expect("structured prompt json");
        assert_eq!(prompt_value, json!({"type":"text","text":"second"}));
        let daemon_messages = req.messages.unwrap();
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
