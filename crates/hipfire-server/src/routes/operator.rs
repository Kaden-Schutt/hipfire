use axum::response::Json;
use hipfire_config::config_schema;
use serde_json::{json, Value};

pub async fn get_config_schema() -> Json<Value> {
    Json(config_schema_json())
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
}
