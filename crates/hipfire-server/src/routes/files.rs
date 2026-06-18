use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Multipart, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::state::{SharedState, StoredFile};

pub async fn list_files(State(state): State<SharedState>) -> impl IntoResponse {
    let files = state.files.lock().await;
    let mut data = files.values().map(file_json).collect::<Vec<_>>();
    data.sort_by_key(|file| file["created_at"].as_u64().unwrap_or_default());
    Json(json!({
        "object": "list",
        "data": data,
    }))
}

pub async fn create_file(State(state): State<SharedState>, mut multipart: Multipart) -> Response {
    let mut filename = None;
    let mut purpose = None;
    let mut content = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let Some(name) = field.name().map(ToOwned::to_owned) else {
            continue;
        };
        if name == "purpose" {
            match field.text().await {
                Ok(value) => purpose = Some(value),
                Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
            }
            continue;
        }
        if name == "file" {
            filename = field.file_name().map(ToOwned::to_owned);
            match field.bytes().await {
                Ok(bytes) => {
                    let text = match String::from_utf8(bytes.to_vec()) {
                        Ok(text) => text,
                        Err(_) => {
                            return error(
                                StatusCode::BAD_REQUEST,
                                "batch files must be UTF-8 JSONL".to_string(),
                            )
                        }
                    };
                    content = Some(text);
                }
                Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
            }
        }
    }

    let Some(content) = content else {
        return error(
            StatusCode::BAD_REQUEST,
            "missing multipart file field".to_string(),
        );
    };
    let filename = filename.unwrap_or_else(|| "batch.jsonl".to_string());
    let purpose = purpose.unwrap_or_else(|| "batch".to_string());
    if purpose != "batch" {
        return error(
            StatusCode::BAD_REQUEST,
            "only purpose=batch is supported".to_string(),
        );
    }

    let file = StoredFile {
        id: format!("file_{}", Uuid::new_v4().simple()),
        filename,
        bytes: content.len(),
        purpose,
        created_at: now_secs(),
        content,
    };
    let body = file_json(&file);
    store_file(&state, file).await;
    Json(body).into_response()
}

pub async fn get_file(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    match state.files.lock().await.get(&id).cloned() {
        Some(file) => Json(file_json(&file)).into_response(),
        None => error(StatusCode::NOT_FOUND, format!("file not found: {id}")),
    }
}

pub async fn get_file_content(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Response {
    match state.files.lock().await.get(&id).cloned() {
        Some(file) => (
            [(header::CONTENT_TYPE, "application/jsonl; charset=utf-8")],
            file.content,
        )
            .into_response(),
        None => error(StatusCode::NOT_FOUND, format!("file not found: {id}")),
    }
}

pub async fn delete_file(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    let deleted = state.files.lock().await.remove(&id).is_some();
    state.file_order.lock().await.retain(|entry| entry != &id);
    Json(json!({
        "id": id,
        "object": "file",
        "deleted": deleted,
    }))
    .into_response()
}

pub(crate) async fn store_generated_file(
    state: &SharedState,
    filename: String,
    content: String,
) -> StoredFile {
    let file = StoredFile {
        id: format!("file_{}", Uuid::new_v4().simple()),
        filename,
        bytes: content.len(),
        purpose: "batch".to_string(),
        created_at: now_secs(),
        content,
    };
    store_file(state, file.clone()).await;
    file
}

pub(crate) fn file_json(file: &StoredFile) -> Value {
    json!({
        "id": file.id,
        "object": "file",
        "bytes": file.bytes,
        "created_at": file.created_at,
        "filename": file.filename,
        "purpose": file.purpose,
    })
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn store_file(state: &SharedState, file: StoredFile) {
    let max = files_state_max();
    if max == 0 {
        return;
    }
    {
        let mut files = state.files.lock().await;
        files.insert(file.id.clone(), file.clone());
    }
    let mut order = state.file_order.lock().await;
    order.retain(|id| id != &file.id);
    order.push_back(file.id);
    while order.len() > max {
        if let Some(evicted) = order.pop_front() {
            state.files.lock().await.remove(&evicted);
        }
    }
}

fn files_state_max() -> usize {
    std::env::var("HIPFIRE_FILES_STATE_MAX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(256)
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
    fn file_json_omits_content() {
        let file = StoredFile {
            id: "file_1".to_string(),
            filename: "input.jsonl".to_string(),
            bytes: 2,
            purpose: "batch".to_string(),
            created_at: 10,
            content: "{}".to_string(),
        };
        let body = file_json(&file);
        assert_eq!(body["id"], "file_1");
        assert!(body.get("content").is_none());
    }
}
