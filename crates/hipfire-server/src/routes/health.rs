use axum::{extract::State, response::Json};
use serde_json::{json, Value};

use crate::state::SharedState;

pub async fn get_health(state: State<SharedState>) -> Json<Value> {
    let loaded = state.loaded_model_path.lock().await.clone();
    Json(json!({
        "status": "ok",
        "model": loaded,
    }))
}
