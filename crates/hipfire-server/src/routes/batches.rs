use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::routes::{
    chat::{
        execute_blocking_chat, openai_chat_completion_response_with_tool_calls_json, ChatRequest,
    },
    files::store_generated_file,
    responses::{execute_responses, ResponsesRequest},
};
use crate::state::{SharedState, StoredBatch};

#[derive(Debug, Deserialize)]
pub struct CreateBatchRequest {
    pub input_file_id: String,
    pub endpoint: String,
    pub completion_window: Option<String>,
}

#[derive(Clone, Debug)]
struct BatchEntry {
    custom_id: String,
    url: String,
    body: Value,
}

#[derive(Clone, Debug, Serialize)]
struct BatchValidationError {
    line: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    custom_id: Option<String>,
    code: String,
    message: String,
}

pub async fn list_batches(State(state): State<SharedState>) -> impl IntoResponse {
    let batches = state.batches.lock().await;
    let mut data = batches.values().map(batch_json).collect::<Vec<_>>();
    data.sort_by_key(|batch| batch["created_at"].as_u64().unwrap_or_default());
    Json(json!({
        "object": "list",
        "data": data,
    }))
}

pub async fn create_batch(
    State(state): State<SharedState>,
    Json(body): Json<CreateBatchRequest>,
) -> Response {
    if !matches!(
        body.endpoint.as_str(),
        "/v1/chat/completions" | "/v1/responses"
    ) {
        return error(
            StatusCode::BAD_REQUEST,
            format!("unsupported batch endpoint: {}", body.endpoint),
        );
    }

    let input_file = match state.files.lock().await.get(&body.input_file_id).cloned() {
        Some(file) => file,
        None => {
            return error(
                StatusCode::NOT_FOUND,
                format!("input file not found: {}", body.input_file_id),
            )
        }
    };
    if input_file.purpose != "batch" {
        return error(
            StatusCode::BAD_REQUEST,
            "input_file_id must reference a purpose=batch file".to_string(),
        );
    }

    let parsed = validate_batch_input_for_endpoint(&input_file.content, &body.endpoint);
    let batch = StoredBatch {
        id: format!("batch_{}", Uuid::new_v4().simple()),
        status: if parsed.errors.is_empty() {
            "in_progress".to_string()
        } else {
            "failed".to_string()
        },
        endpoint: body.endpoint.clone(),
        completion_window: body.completion_window.unwrap_or_else(|| "24h".to_string()),
        input_file_id: body.input_file_id,
        output_file_id: None,
        error_file_id: None,
        request_count: parsed.entries.len() + parsed.errors.len(),
        completed_requests: 0,
        created_at: now_secs(),
        in_progress_at: parsed.errors.is_empty().then_some(now_secs()),
        completed_at: None,
        failed_reason: (!parsed.errors.is_empty()).then_some("batch validation failed".to_string()),
    };
    let batch_id = batch.id.clone();
    store_batch(&state, batch.clone()).await;

    if parsed.errors.is_empty() {
        tokio::spawn(run_batch(state.clone(), batch_id, parsed.entries));
    } else {
        let error_jsonl = batch_error_jsonl(&parsed.errors);
        let error_file =
            store_generated_file(&state, format!("{}_errors.jsonl", batch.id), error_jsonl).await;
        update_batch(&state, &batch.id, |batch| {
            batch.error_file_id = Some(error_file.id);
            batch.completed_at = Some(now_secs());
        })
        .await;
    }

    match state.batches.lock().await.get(&batch.id).cloned() {
        Some(batch) => Json(batch_json(&batch)).into_response(),
        None => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "batch disappeared".to_string(),
        ),
    }
}

pub async fn get_batch(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    match state.batches.lock().await.get(&id).cloned() {
        Some(batch) => Json(batch_json(&batch)).into_response(),
        None => error(StatusCode::NOT_FOUND, format!("batch not found: {id}")),
    }
}

pub async fn cancel_batch(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    let mut found = false;
    update_batch(&state, &id, |batch| {
        found = true;
        if !matches!(batch.status.as_str(), "completed" | "failed" | "cancelled") {
            batch.status = "cancelled".to_string();
            batch.completed_at = Some(now_secs());
        }
    })
    .await;

    if !found {
        return error(StatusCode::NOT_FOUND, format!("batch not found: {id}"));
    }
    get_batch(State(state), Path(id)).await
}

async fn run_batch(state: SharedState, batch_id: String, entries: Vec<BatchEntry>) {
    let mut output_lines = Vec::new();
    let mut completed = 0;

    for entry in entries {
        let cancelled = state
            .batches
            .lock()
            .await
            .get(&batch_id)
            .map(|batch| batch.status == "cancelled")
            .unwrap_or(true);
        if cancelled {
            return;
        }

        let line = match execute_batch_entry(state.clone(), entry.clone()).await {
            Ok(body) => json!({
                "custom_id": entry.custom_id,
                "response": {
                    "status_code": 200,
                    "body": body,
                },
            }),
            Err(error_body) => json!({
                "custom_id": entry.custom_id,
                "error": error_body.get("error").cloned().unwrap_or(error_body),
            }),
        };
        output_lines.push(serde_json::to_string(&line).expect("batch output line is JSON"));
        completed += 1;
        update_batch(&state, &batch_id, |batch| {
            batch.completed_requests = completed;
        })
        .await;
    }

    let content = if output_lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", output_lines.join("\n"))
    };
    let output_file =
        store_generated_file(&state, format!("{batch_id}_output.jsonl"), content).await;
    update_batch(&state, &batch_id, |batch| {
        if batch.status != "cancelled" {
            batch.status = "completed".to_string();
            batch.output_file_id = Some(output_file.id);
            batch.completed_at = Some(now_secs());
        }
    })
    .await;
}

async fn execute_batch_entry(state: SharedState, entry: BatchEntry) -> Result<Value, Value> {
    let mut body = entry.body;
    if let Some(obj) = body.as_object_mut() {
        obj.insert("stream".to_string(), Value::Bool(false));
    }

    match entry.url.as_str() {
        "/v1/chat/completions" => {
            let request: ChatRequest = serde_json::from_value(body).map_err(
                |e| json!({"error": {"message": e.to_string(), "type": "invalid_request_error"}}),
            )?;
            let result = execute_blocking_chat(state, request).await?;
            Ok(openai_chat_completion_response_with_tool_calls_json(
                &result.req_id,
                result.created,
                &result.model,
                &result.text,
                &result.tool_calls,
                &result.done,
                result.request_max_tokens,
            ))
        }
        "/v1/responses" => {
            let request: ResponsesRequest = serde_json::from_value(body).map_err(
                |e| json!({"error": {"message": e.to_string(), "type": "invalid_request_error"}}),
            )?;
            execute_responses(state, request).await
        }
        _ => Err(json!({
            "error": {
                "message": format!("unsupported batch endpoint: {}", entry.url),
                "type": "invalid_request_error",
            }
        })),
    }
}

struct ParsedBatchInput {
    entries: Vec<BatchEntry>,
    errors: Vec<BatchValidationError>,
}

fn validate_batch_input_for_endpoint(raw: &str, expected_endpoint: &str) -> ParsedBatchInput {
    if raw.trim().is_empty() {
        return ParsedBatchInput {
            entries: Vec::new(),
            errors: vec![BatchValidationError {
                line: 1,
                custom_id: None,
                code: "empty_batch".to_string(),
                message: "batch file is empty".to_string(),
            }],
        };
    }

    let mut entries = Vec::new();
    let mut errors = Vec::new();
    let mut seen = HashSet::new();
    let mut expected_model: Option<String> = None;
    for (idx, line) in raw.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed: Value = match serde_json::from_str(trimmed) {
            Ok(value) => value,
            Err(_) => {
                errors.push(validation_error(
                    line_no,
                    None,
                    "invalid_json",
                    "line is not valid JSON",
                ));
                continue;
            }
        };
        let Some(obj) = parsed.as_object() else {
            errors.push(validation_error(
                line_no,
                None,
                "invalid_object",
                "line must be a JSON object",
            ));
            continue;
        };
        let custom_id = obj
            .get("custom_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        let Some(custom_id) = custom_id.filter(|id| !id.trim().is_empty()) else {
            errors.push(validation_error(
                line_no,
                None,
                "invalid_custom_id",
                "missing custom_id",
            ));
            continue;
        };
        if !seen.insert(custom_id.clone()) {
            errors.push(validation_error(
                line_no,
                Some(custom_id.clone()),
                "duplicate_custom_id",
                "duplicate custom_id",
            ));
            continue;
        }
        if obj.get("method").and_then(Value::as_str) != Some("POST") {
            errors.push(validation_error(
                line_no,
                Some(custom_id.clone()),
                "invalid_method",
                "method must be POST",
            ));
        }
        let url = obj.get("url").and_then(Value::as_str).unwrap_or_default();
        if !matches!(url, "/v1/chat/completions" | "/v1/responses") {
            errors.push(validation_error(
                line_no,
                Some(custom_id.clone()),
                "invalid_url",
                "url must be /v1/chat/completions or /v1/responses",
            ));
            continue;
        }
        if url != expected_endpoint {
            errors.push(validation_error(
                line_no,
                Some(custom_id.clone()),
                "endpoint_mismatch",
                "line endpoint does not match batch endpoint",
            ));
            continue;
        }
        let Some(body) = obj.get("body").cloned().filter(Value::is_object) else {
            errors.push(validation_error(
                line_no,
                Some(custom_id.clone()),
                "invalid_body",
                "body must be a JSON object",
            ));
            continue;
        };
        if body.get("stream").and_then(Value::as_bool) == Some(true) {
            errors.push(validation_error(
                line_no,
                Some(custom_id.clone()),
                "streaming_unsupported",
                "stream=true is unsupported in batch mode",
            ));
            continue;
        }
        if body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .is_empty()
        {
            errors.push(validation_error(
                line_no,
                Some(custom_id.clone()),
                "model_missing",
                "batch entries must specify model",
            ));
            continue;
        }
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if let Some(expected) = &expected_model {
            if &model != expected {
                errors.push(validation_error(
                    line_no,
                    Some(custom_id.clone()),
                    "model_mismatch",
                    "model family differs from earlier entries",
                ));
                continue;
            }
        } else {
            expected_model = Some(model);
        }
        let mut body_errors = validate_batch_body_for_url(&body, url, &custom_id);
        if !body_errors.is_empty() {
            errors.append(&mut body_errors);
            continue;
        }
        entries.push(BatchEntry {
            custom_id,
            url: url.to_string(),
            body,
        });
    }

    ParsedBatchInput { entries, errors }
}

fn validate_batch_body_for_url(
    body: &Value,
    url: &str,
    custom_id: &str,
) -> Vec<BatchValidationError> {
    let mut errors = Vec::new();
    if has_non_empty_tools(body) {
        errors.push(validation_error(
            0,
            Some(custom_id.to_string()),
            "tools_unsupported",
            "tools are unsupported in batch mode",
        ));
    }
    match url {
        "/v1/chat/completions" => {
            errors.extend(validate_chat_messages_for_batch(body, custom_id));
        }
        "/v1/responses" => {
            errors.extend(validate_responses_input_for_batch(body, custom_id));
        }
        _ => {}
    }
    errors
}

fn has_non_empty_tools(body: &Value) -> bool {
    body.get("tools")
        .and_then(Value::as_array)
        .map(|tools| !tools.is_empty())
        .unwrap_or(false)
}

fn validate_chat_messages_for_batch(body: &Value, custom_id: &str) -> Vec<BatchValidationError> {
    let valid = body
        .get("messages")
        .and_then(Value::as_array)
        .map(|messages| {
            !messages.is_empty()
                && messages.iter().all(|message| {
                    let Some(obj) = message.as_object() else {
                        return false;
                    };
                    let role_ok = obj
                        .get("role")
                        .and_then(Value::as_str)
                        .map(|role| {
                            matches!(
                                role,
                                "system"
                                    | "developer"
                                    | "user"
                                    | "assistant"
                                    | "tool"
                                    | "toolResult"
                                    | "tool_result"
                            )
                        })
                        .unwrap_or(false);
                    role_ok
                        && obj
                            .get("content")
                            .map(is_supported_batch_content)
                            .unwrap_or(true)
                })
        })
        .unwrap_or(false);

    if valid {
        Vec::new()
    } else {
        vec![validation_error(
            0,
            Some(custom_id.to_string()),
            "invalid_messages",
            "contains invalid chat message entries",
        )]
    }
}

fn validate_responses_input_for_batch(body: &Value, custom_id: &str) -> Vec<BatchValidationError> {
    let Some(input) = body.get("input") else {
        return vec![validation_error(
            0,
            Some(custom_id.to_string()),
            "invalid_responses_input",
            "missing valid responses input",
        )];
    };
    match input {
        Value::String(_) => Vec::new(),
        Value::Array(messages) => validate_responses_messages(messages, custom_id, "input"),
        Value::Object(obj) => match obj.get("messages").and_then(Value::as_array) {
            Some(messages) => validate_responses_messages(messages, custom_id, "input.messages"),
            None => vec![validation_error(
                0,
                Some(custom_id.to_string()),
                "invalid_responses_input",
                "responses object input must include messages array",
            )],
        },
        _ => vec![validation_error(
            0,
            Some(custom_id.to_string()),
            "invalid_responses_input",
            "missing valid responses input",
        )],
    }
}

fn validate_responses_messages(
    messages: &[Value],
    custom_id: &str,
    field: &str,
) -> Vec<BatchValidationError> {
    if messages.is_empty() {
        return vec![validation_error(
            0,
            Some(custom_id.to_string()),
            "invalid_responses_input",
            &format!("{field} must contain at least one message"),
        )];
    }
    let mut errors = Vec::new();
    for message in messages {
        let Some(obj) = message.as_object() else {
            errors.push(validation_error(
                0,
                Some(custom_id.to_string()),
                "invalid_responses_input",
                &format!("includes non-message entries in {field}"),
            ));
            break;
        };
        let content = obj.get("content").unwrap_or(&Value::Null);
        if has_unsupported_batch_content(content) {
            errors.push(validation_error(
                0,
                Some(custom_id.to_string()),
                "unsupported_content",
                &format!("includes unsupported content in {field}"),
            ));
            break;
        }
        if obj.get("role").and_then(Value::as_str).is_none() || !is_supported_batch_content(content)
        {
            errors.push(validation_error(
                0,
                Some(custom_id.to_string()),
                "invalid_responses_input",
                &format!("includes non-message entries in {field}"),
            ));
            break;
        }
    }
    errors
}

fn is_supported_batch_content(content: &Value) -> bool {
    match content {
        Value::String(_) | Value::Null => true,
        Value::Array(parts) => parts.iter().all(|part| {
            part.as_str().is_some()
                || part
                    .as_object()
                    .map(|obj| {
                        matches!(
                            obj.get("type").and_then(Value::as_str),
                            Some("text" | "input_text" | "output_text") | None
                        ) && obj.get("image_url").is_none()
                    })
                    .unwrap_or(false)
        }),
        _ => false,
    }
}

fn has_unsupported_batch_content(content: &Value) -> bool {
    match content {
        Value::Array(parts) => parts.iter().any(|part| {
            part.as_object()
                .map(|obj| {
                    obj.get("image_url").is_some()
                        || matches!(obj.get("type").and_then(Value::as_str), Some("image_url"))
                })
                .unwrap_or(false)
        }),
        _ => false,
    }
}

fn validation_error(
    line: usize,
    custom_id: Option<String>,
    code: &str,
    message: &str,
) -> BatchValidationError {
    BatchValidationError {
        line,
        custom_id,
        code: code.to_string(),
        message: message.to_string(),
    }
}

fn batch_error_jsonl(errors: &[BatchValidationError]) -> String {
    errors
        .iter()
        .map(|err| {
            serde_json::to_string(&json!({
                "custom_id": err.custom_id.clone().unwrap_or_else(|| format!("line-{}", err.line)),
                "processing_status": "rejected",
                "error": {
                    "code": err.code,
                    "message": err.message,
                    "type": "invalid_request_error",
                }
            }))
            .expect("batch error line is JSON")
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn batch_json(batch: &StoredBatch) -> Value {
    json!({
        "id": batch.id,
        "object": "batch",
        "status": batch.status,
        "endpoint": batch.endpoint,
        "completion_window": batch.completion_window,
        "input_file_id": batch.input_file_id,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "request_counts": {
            "total": batch.request_count,
            "completed": batch.completed_requests,
            "failed": if batch.status == "failed" { batch.request_count } else { 0 },
        },
        "request_count": batch.request_count,
        "completed_requests": batch.completed_requests,
        "created_at": batch.created_at,
        "in_progress_at": batch.in_progress_at,
        "completed_at": batch.completed_at,
        "failed_reason": batch.failed_reason,
    })
}

async fn store_batch(state: &SharedState, batch: StoredBatch) {
    let max = batches_state_max();
    if max == 0 {
        return;
    }
    {
        let mut batches = state.batches.lock().await;
        batches.insert(batch.id.clone(), batch.clone());
    }
    let mut order = state.batch_order.lock().await;
    order.retain(|id| id != &batch.id);
    order.push_back(batch.id);
    while order.len() > max {
        if let Some(evicted) = order.pop_front() {
            state.batches.lock().await.remove(&evicted);
        }
    }
}

async fn update_batch<F>(state: &SharedState, batch_id: &str, update: F)
where
    F: FnOnce(&mut StoredBatch),
{
    if let Some(batch) = state.batches.lock().await.get_mut(batch_id) {
        update(batch);
    }
}

fn batches_state_max() -> usize {
    std::env::var("HIPFIRE_BATCHES_STATE_MAX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(128)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn error(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_batch_endpoint_mismatch() {
        let raw = r#"{"custom_id":"one","method":"POST","url":"/v1/responses","body":{"model":"qwen","input":"hi"}}"#;
        let parsed = validate_batch_input_for_endpoint(raw, "/v1/chat/completions");
        assert_eq!(parsed.entries.len(), 0);
        assert_eq!(parsed.errors[0].code, "endpoint_mismatch");
    }

    #[test]
    fn validates_chat_batch_entry() {
        let raw = r#"{"custom_id":"one","method":"POST","url":"/v1/chat/completions","body":{"model":"qwen","messages":[{"role":"user","content":"hi"}]}}"#;
        let parsed = validate_batch_input_for_endpoint(raw, "/v1/chat/completions");
        assert_eq!(parsed.entries.len(), 1);
        assert!(parsed.errors.is_empty());
    }

    #[test]
    fn rejects_tools_in_batch_mode() {
        let raw = r#"{"custom_id":"one","method":"POST","url":"/v1/chat/completions","body":{"model":"qwen","tools":[{"type":"function"}],"messages":[{"role":"user","content":"hi"}]}}"#;
        let parsed = validate_batch_input_for_endpoint(raw, "/v1/chat/completions");
        assert_eq!(parsed.entries.len(), 0);
        assert_eq!(parsed.errors[0].code, "tools_unsupported");
    }

    #[test]
    fn rejects_invalid_chat_messages_in_batch_mode() {
        let raw = r#"{"custom_id":"one","method":"POST","url":"/v1/chat/completions","body":{"model":"qwen","messages":[{"role":"bad","content":"hi"}]}}"#;
        let parsed = validate_batch_input_for_endpoint(raw, "/v1/chat/completions");
        assert_eq!(parsed.entries.len(), 0);
        assert_eq!(parsed.errors[0].code, "invalid_messages");
    }

    #[test]
    fn rejects_unsupported_responses_content_in_batch_mode() {
        let raw = r#"{"custom_id":"one","method":"POST","url":"/v1/responses","body":{"model":"qwen","input":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}}"#;
        let parsed = validate_batch_input_for_endpoint(raw, "/v1/responses");
        assert_eq!(parsed.entries.len(), 0);
        assert_eq!(parsed.errors[0].code, "unsupported_content");
    }

    #[test]
    fn rejects_model_mismatch_in_batch_mode() {
        let raw = r#"{"custom_id":"one","method":"POST","url":"/v1/chat/completions","body":{"model":"qwen-a","messages":[{"role":"user","content":"hi"}]}}
{"custom_id":"two","method":"POST","url":"/v1/chat/completions","body":{"model":"qwen-b","messages":[{"role":"user","content":"hi"}]}}"#;
        let parsed = validate_batch_input_for_endpoint(raw, "/v1/chat/completions");
        assert_eq!(parsed.entries.len(), 1);
        assert_eq!(parsed.errors[0].code, "model_mismatch");
    }

    #[test]
    fn batch_json_includes_openai_counts() {
        let batch = StoredBatch {
            id: "batch_1".to_string(),
            status: "completed".to_string(),
            endpoint: "/v1/chat/completions".to_string(),
            completion_window: "24h".to_string(),
            input_file_id: "file_1".to_string(),
            output_file_id: Some("file_2".to_string()),
            error_file_id: None,
            request_count: 1,
            completed_requests: 1,
            created_at: 10,
            in_progress_at: Some(10),
            completed_at: Some(11),
            failed_reason: None,
        };
        let body = batch_json(&batch);
        assert_eq!(body["request_counts"]["total"], 1);
        assert_eq!(body["output_file_id"], "file_2");
    }

    #[test]
    fn failed_batch_counts_rejected_lines() {
        let batch = StoredBatch {
            id: "batch_1".to_string(),
            status: "failed".to_string(),
            endpoint: "/v1/chat/completions".to_string(),
            completion_window: "24h".to_string(),
            input_file_id: "file_1".to_string(),
            output_file_id: None,
            error_file_id: Some("file_err".to_string()),
            request_count: 2,
            completed_requests: 0,
            created_at: 10,
            in_progress_at: None,
            completed_at: Some(11),
            failed_reason: Some("batch validation failed".to_string()),
        };
        let body = batch_json(&batch);
        assert_eq!(body["request_counts"]["total"], 2);
        assert_eq!(body["request_counts"]["failed"], 2);
    }
}
