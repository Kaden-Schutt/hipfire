use axum::{extract::State, response::Json};
use hipfire_model::model_display_name;
use serde_json::Value;
use std::path::Path;

use crate::model::discovery::list_local_models;
use crate::state::SharedState;

pub async fn get_models(_state: State<SharedState>) -> Json<Value> {
    let models = list_local_models();
    Json(bun_model_list_json(models.iter()))
}

fn bun_model_list_json<I, P>(models: I) -> Value
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    let data: Vec<Value> = models
        .into_iter()
        .map(|path| {
            serde_json::json!({
                "id": model_display_name(path.as_ref()),
            })
        })
        .collect();

    serde_json::json!({ "data": data })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn model_list_matches_bun_shape() {
        let models = [
            PathBuf::from("/models/qwen3.5-9b-mq4.hfq"),
            PathBuf::from("/models/qwen3.5-9b-q8.hfq"),
        ];

        assert_eq!(
            bun_model_list_json(models.iter()),
            serde_json::json!({
                "data": [
                    { "id": "qwen3.5-9b-mq4" },
                    { "id": "qwen3.5-9b-q8" }
                ]
            })
        );
    }
}
