use axum::{extract::State, response::Json};
#[cfg(test)]
use hipfire_model::model_display_name;
use hipfire_model::LlmModelRegistryEntry;
use serde_json::Value;
#[cfg(test)]
use std::path::Path;

use crate::model::discovery::local_llm_registry;
use crate::state::SharedState;

pub async fn get_models(_state: State<SharedState>) -> Json<Value> {
    let registry = local_llm_registry();
    Json(model_registry_openai_json(registry.models.iter()))
}

pub async fn get_model_registry(_state: State<SharedState>) -> Json<Value> {
    Json(
        serde_json::to_value(local_llm_registry()).unwrap_or_else(|err| {
            serde_json::json!({
                "error": {
                    "message": format!("failed to serialize model registry: {err}"),
                    "type": "internal_error"
                }
            })
        }),
    )
}

#[cfg(test)]
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

fn model_registry_openai_json<'a, I>(models: I) -> Value
where
    I: IntoIterator<Item = &'a LlmModelRegistryEntry>,
{
    let data: Vec<Value> = models
        .into_iter()
        .map(|model| {
            serde_json::json!({
                "id": model.id,
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

    #[test]
    fn registry_model_list_preserves_openai_compatible_shape() {
        let models = [LlmModelRegistryEntry {
            id: "qwen3.5-9b-mq4".to_string(),
            file: "qwen3.5-9b-mq4.hfq".to_string(),
            path: "/models/qwen3.5-9b-mq4.hfq".to_string(),
            bytes: 4,
            model: "qwen3.5".to_string(),
            size: Some("9b".to_string()),
            tags: Vec::new(),
            features: Vec::new(),
            quant: "mq4".to_string(),
            arch: None,
            triattn: Vec::new(),
            drafts: Vec::new(),
            chat_templates: Vec::new(),
        }];

        assert_eq!(
            model_registry_openai_json(models.iter()),
            serde_json::json!({
                "data": [
                    { "id": "qwen3.5-9b-mq4" }
                ]
            })
        );
    }
}
