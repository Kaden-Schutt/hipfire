use axum::{extract::State, response::Json};
use hipfire_model::openai_model_list_json;
use serde_json::Value;

use crate::model::discovery::list_local_models;
use crate::state::SharedState;

pub async fn get_models(_state: State<SharedState>) -> Json<Value> {
    let models = list_local_models();
    Json(openai_model_list_json(models.iter()))
}
