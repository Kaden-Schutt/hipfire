use axum::{extract::Query, response::Json};
use hipfire_config::{
    config_layers_from_document, config_path, config_schema, resolve_config_layers,
};
use serde::Deserialize;
use serde_json::{json, Value};

pub async fn get_config_schema() -> Json<Value> {
    Json(config_schema_json())
}

#[derive(Debug, Deserialize)]
pub struct ResolvedConfigQuery {
    pub model: Option<String>,
}

pub async fn get_resolved_config(Query(query): Query<ResolvedConfigQuery>) -> Json<Value> {
    let path = config_path();
    let (document, read_error) = match std::fs::read_to_string(&path) {
        Ok(raw) => match serde_json::from_str::<Value>(&raw) {
            Ok(value) => (value, None),
            Err(err) => (json!({}), Some(format!("parse error: {err}"))),
        },
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => (json!({}), None),
        Err(err) => (json!({}), Some(format!("read error: {err}"))),
    };

    Json(resolved_config_json(
        &document,
        query.model.as_deref(),
        Some(path.display().to_string()),
        read_error,
    ))
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

fn resolved_config_json(
    document: &Value,
    model: Option<&str>,
    config_path: Option<String>,
    read_error: Option<String>,
) -> Value {
    let layers = config_layers_from_document(document, model);
    let resolution = resolve_config_layers(config_schema(), &layers);
    json!({
        "config_path": config_path,
        "model": model,
        "read_error": read_error,
        "layers": layers,
        "resolution": resolution,
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

        let payload = resolved_config_json(&document, Some("qwen3.5:9b"), None, None);
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
}
