use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use hipfire_operator::training::{
    list_training_runs, load_training_run_detail, read_training_run_events, TrainingRunList,
};
use serde_json::json;

use crate::state::SharedState;

const RECENT_EVENT_LIMIT: usize = 200;

pub async fn list_training_runs_route(State(state): State<SharedState>) -> Json<TrainingRunList> {
    Json(list_training_runs(&state.training_runs_dir))
}

pub async fn get_training_run(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Response {
    match load_training_run_detail(&state.training_runs_dir, &id, RECENT_EVENT_LIMIT) {
        Some(detail) => Json(detail).into_response(),
        None => not_found(&id),
    }
}

pub async fn get_training_run_events(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Response {
    let run_dir = state.training_runs_dir.join(&id);
    if !run_dir.is_dir() {
        return not_found(&id);
    }
    Json(read_training_run_events(&run_dir, &id, None)).into_response()
}

fn not_found(id: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "message": format!("training run '{id}' not found"),
                "type": "not_found"
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use hipfire_config::LoadedConfig;
    use std::{fs, path::PathBuf};

    struct TempTree {
        path: PathBuf,
    }

    impl TempTree {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "hipfire-server-training-{name}-{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn state_with_runs(path: PathBuf) -> SharedState {
        crate::state::AppState::new_loaded_with_training_runs_dir(
            LoadedConfig::from_config(Default::default()),
            path,
        )
    }

    fn write_run(root: &std::path::Path, id: &str, status: &str) -> PathBuf {
        let dir = root.join(id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("status.json"),
            format!(
                r#"{{"id":"{id}","kind":"drafter","status":"{status}","updated_at":"2026-06-20T10:00:00Z","metrics":{{"best_eval_metric":0.42}}}}"#
            ),
        )
        .unwrap();
        dir
    }

    async fn response_json(response: Response) -> serde_json::Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn list_route_returns_empty_runs_for_missing_dir() {
        let tmp = TempTree::new("empty");
        let state = state_with_runs(tmp.path.join("missing"));
        let payload = list_training_runs_route(State(state)).await.0;
        assert!(payload.runs.is_empty());
        assert!(payload.errors.is_empty());
    }

    #[tokio::test]
    async fn detail_route_returns_run_and_recent_events() {
        let tmp = TempTree::new("detail");
        let run_dir = write_run(&tmp.path, "run-a", "training");
        fs::write(
            run_dir.join("events.jsonl"),
            "{\"type\":\"run_started\"}\n{\"type\":\"train_progress\",\"epoch\":1}\n",
        )
        .unwrap();
        let state = state_with_runs(tmp.path.clone());
        let response = get_training_run(State(state), Path("run-a".to_string())).await;
        assert_eq!(response.status(), StatusCode::OK);
        let json = response_json(response).await;
        assert_eq!(json["summary"]["id"], "run-a");
        assert_eq!(json["recent_events"][1]["event"]["type"], "train_progress");
    }

    #[tokio::test]
    async fn events_route_preserves_malformed_lines() {
        let tmp = TempTree::new("events");
        let run_dir = write_run(&tmp.path, "run-b", "failed");
        fs::write(
            run_dir.join("events.jsonl"),
            "{\"type\":\"error\",\"message\":\"boom\"}\nnot-json\n",
        )
        .unwrap();
        let state = state_with_runs(tmp.path.clone());
        let response = get_training_run_events(State(state), Path("run-b".to_string())).await;
        assert_eq!(response.status(), StatusCode::OK);
        let json = response_json(response).await;
        assert_eq!(json["events"][0]["event"]["type"], "error");
        assert_eq!(json["errors"][0]["line"], 2);
    }

    #[tokio::test]
    async fn missing_run_returns_404() {
        let tmp = TempTree::new("missing");
        let state = state_with_runs(tmp.path.clone());
        let response = get_training_run(State(state), Path("missing".to_string())).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[allow(dead_code)]
    fn _body(_: Body) {}
}
