use axum::{
    extract::{Query, State},
    response::Json,
};
use hipfire_config::{config_schema, LoadedConfig};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::SharedState;

pub async fn get_config_schema() -> Json<Value> {
    Json(config_schema_json())
}

#[derive(Debug, Deserialize)]
pub struct ResolvedConfigQuery {
    pub model: Option<String>,
}

pub async fn get_resolved_config(
    State(state): State<SharedState>,
    Query(query): Query<ResolvedConfigQuery>,
) -> Json<Value> {
    let loaded = state.loaded_config.lock().await;
    Json(resolved_config_json(&loaded, query.model.as_deref()))
}

fn config_schema_json() -> Value {
    serde_json::to_value(config_schema()).unwrap_or_else(|err| {
        json!({
            "error": {
                "message": format!("failed to serialize config schema: {err}"),
                "type": "internal_error"
            }
        })
    })
}

fn resolved_config_json(loaded: &LoadedConfig, model: Option<&str>) -> Value {
    let (config, layers, resolution, diagnostics) = match model {
        Some(model) => {
            let resolved = loaded.resolve_for_model(model);
            (
                resolved.config,
                resolved.layers,
                resolved.resolution,
                resolved.diagnostics,
            )
        }
        None => (
            loaded.config.clone(),
            loaded.layers.clone(),
            loaded.resolution.clone(),
            loaded.diagnostics.clone(),
        ),
    };
    json!({
        "source": "active_runtime",
        "config_path": loaded.config_path.display().to_string(),
        "model": model,
        "read_error": loaded.read_error.clone(),
        "diagnostics": diagnostics,
        "layers": layers,
        "resolution": resolution,
        "config": config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_schema_route_exposes_schema_fields() {
        let payload = config_schema_json();
        let fields = payload.as_array().expect("schema array");

        assert!(fields
            .iter()
            .any(|field| field.get("key").and_then(Value::as_str) == Some("max_tokens")));
        assert!(fields.iter().any(|field| {
            field
                .get("requirement")
                .and_then(|req| req.get("kind"))
                .and_then(Value::as_str)
                == Some("required_when")
        }));
    }

    #[test]
    fn resolved_config_route_explains_model_override_source() {
        let document = json!({
            "max_tokens": 256,
            "model_overrides": {
                "qwen3.5:9b": {
                    "max_tokens": 64
                }
            }
        });

        let loaded = hipfire_config::loaded_config_from_document(
            std::path::PathBuf::from("/tmp/config.json"),
            document,
            None,
            Vec::new(),
        );

        let payload = resolved_config_json(&loaded, Some("qwen3.5:9b"));
        let values = payload["resolution"]["values"]
            .as_array()
            .expect("resolved values");
        let max_tokens = values
            .iter()
            .find(|value| value["key"] == "max_tokens")
            .expect("max_tokens");

        assert_eq!(max_tokens["value"], json!(64));
        assert_eq!(max_tokens["source"]["kind"], "model");
        assert_eq!(max_tokens["source"]["id"], "qwen3.5:9b");
        assert!(max_tokens["overrode"]
            .as_array()
            .expect("overrode")
            .iter()
            .any(|source| source["kind"] == "global"));
    }

    #[test]
    fn resolved_config_route_reports_active_cli_layer() {
        let document = json!({
            "host": "127.0.0.1",
            "port": 11435
        });
        let cli_layer = hipfire_config::ConfigLayer::new(hipfire_config::ConfigLayerKind::Cli)
            .with_value("port", 12000);
        let loaded = hipfire_config::loaded_config_from_document(
            std::path::PathBuf::from("/tmp/config.json"),
            document,
            None,
            vec![cli_layer],
        );

        let payload = resolved_config_json(&loaded, None);
        assert_eq!(payload["config"]["port"], json!(12000));
        assert!(payload["layers"]
            .as_array()
            .expect("layers")
            .iter()
            .any(|layer| layer["kind"] == "cli"));
    }
}
