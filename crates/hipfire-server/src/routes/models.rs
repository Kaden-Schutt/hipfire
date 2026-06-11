use axum::{extract::State, response::Json};
use serde_json::{json, Value};

use crate::model::discovery::{list_local_models, model_display_name};
use crate::state::SharedState;

pub async fn get_models(_state: State<SharedState>) -> Json<Value> {
    let models = list_local_models();
    let data: Vec<Value> = models
        .iter()
        .map(|p| json!({ "id": model_display_name(p), "object": "model" }))
        .collect();
    Json(json!({ "object": "list", "data": data }))
}
