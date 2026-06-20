use axum::{
    extract::{Query, State},
    response::{Html, Json},
};
use hipfire_config::{config_schema, LoadedConfig};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::SharedState;

pub async fn get_operator_index() -> Html<&'static str> {
    Html(OPERATOR_INDEX_HTML)
}

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
        "host_config_path": loaded.host_config_path.display().to_string(),
        "model": model,
        "read_error": loaded.read_error.clone(),
        "host_read_error": loaded.host_read_error.clone(),
        "diagnostics": diagnostics,
        "layers": layers,
        "resolution": resolution,
        "config": config,
    })
}

const OPERATOR_INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>hipfire operator</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #172026;
      --muted: #66717c;
      --line: #d7dde3;
      --accent: #0f766e;
      --accent-2: #7c3aed;
      --warn: #b45309;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111417;
        --panel: #171b20;
        --text: #e7ecef;
        --muted: #9aa5af;
        --line: #29313a;
        --accent: #2dd4bf;
        --accent-2: #a78bfa;
        --warn: #f59e0b;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      width: min(1280px, 100%);
      margin: 0 auto;
      padding: 18px 24px 32px;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: end;
      gap: 12px;
      padding-bottom: 16px;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    input, button {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
    }
    input {
      width: min(360px, 76vw);
      padding: 0 10px;
    }
    button {
      padding: 0 12px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    .status {
      margin-left: auto;
      color: var(--muted);
      min-width: 180px;
      text-align: right;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px 12px;
      min-width: 0;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    .metric strong {
      display: block;
      margin-top: 5px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 14px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
    }
    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      background: var(--panel);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      z-index: 1;
    }
    td.key { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--accent); }
    td.value { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; max-width: 280px; overflow-wrap: anywhere; }
    .source { color: var(--accent-2); }
    .muted { color: var(--muted); }
    .warn { color: var(--warn); }
    @media (max-width: 820px) {
      header, main { padding-left: 14px; padding-right: 14px; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .status { width: 100%; text-align: left; margin-left: 0; }
      table { font-size: 13px; }
      th:nth-child(4), td:nth-child(4), th:nth-child(5), td:nth-child(5) { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <h1>hipfire operator</h1>
    <div id="status" class="status">connecting</div>
  </header>
  <main>
    <div class="toolbar">
      <label>Model
        <input id="model" name="model" autocomplete="off" placeholder="optional model tag">
      </label>
      <button id="refresh" type="button">Refresh</button>
    </div>
    <section class="summary" aria-label="Config summary">
      <div class="metric"><span>Source</span><strong id="source">-</strong></div>
      <div class="metric"><span>Path</span><strong id="path">-</strong></div>
      <div class="metric"><span>Fields</span><strong id="fields">-</strong></div>
      <div class="metric"><span>Diagnostics</span><strong id="diagnostics">-</strong></div>
    </section>
    <table>
      <thead>
        <tr>
          <th>Key</th>
          <th>Value</th>
          <th>Source</th>
          <th>Scope</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const modelEl = document.getElementById("model");
    const refreshEl = document.getElementById("refresh");
    const rowsEl = document.getElementById("rows");
    const sourceEl = document.getElementById("source");
    const pathEl = document.getElementById("path");
    const fieldsEl = document.getElementById("fields");
    const diagnosticsEl = document.getElementById("diagnostics");

    function text(value) {
      if (value === null || value === undefined) return "";
      if (typeof value === "string") return value;
      return JSON.stringify(value);
    }

    function sourceLabel(source) {
      if (!source) return "";
      return source.id ? `${source.kind}:${source.id}` : source.kind;
    }

    async function loadConfig() {
      const model = modelEl.value.trim();
      const suffix = model ? `?model=${encodeURIComponent(model)}` : "";
      statusEl.textContent = "loading";
      const [schemaResp, resolvedResp] = await Promise.all([
        fetch("/operator/config/schema"),
        fetch(`/operator/config/resolved${suffix}`),
      ]);
      if (!schemaResp.ok) throw new Error(`schema ${schemaResp.status}`);
      if (!resolvedResp.ok) throw new Error(`resolved ${resolvedResp.status}`);
      const schema = await schemaResp.json();
      const resolved = await resolvedResp.json();
      render(schema, resolved);
      statusEl.textContent = model ? `model ${model}` : "active runtime";
    }

    function render(schema, resolved) {
      const fields = new Map(schema.map((field) => [field.key, field]));
      const values = resolved.resolution.values || [];
      sourceEl.textContent = resolved.source || "-";
      pathEl.textContent = resolved.config_path || "-";
      fieldsEl.textContent = String(values.length);
      const diagnostics = resolved.diagnostics || [];
      const readError = resolved.read_error ? [resolved.read_error] : [];
      diagnosticsEl.textContent = [...readError, ...diagnostics.map((d) => d.message)].join("; ") || "none";
      diagnosticsEl.className = diagnostics.length || readError.length ? "warn" : "";
      rowsEl.replaceChildren(...values.map((entry) => {
        const field = fields.get(entry.key) || {};
        const tr = document.createElement("tr");
        const key = document.createElement("td");
        key.className = "key";
        key.textContent = entry.key;
        const value = document.createElement("td");
        value.className = "value";
        value.textContent = text(entry.value);
        const source = document.createElement("td");
        source.className = entry.missing_required ? "warn" : "source";
        source.textContent = entry.missing_required ? "required" : sourceLabel(entry.source);
        const scope = document.createElement("td");
        scope.className = "muted";
        scope.textContent = (field.scopes || []).join(", ");
        const desc = document.createElement("td");
        desc.textContent = field.description || "";
        tr.append(key, value, source, scope, desc);
        return tr;
      }));
    }

    refreshEl.addEventListener("click", () => loadConfig().catch(showError));
    modelEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") loadConfig().catch(showError);
    });
    function showError(error) {
      statusEl.textContent = error.message;
      statusEl.className = "status warn";
    }
    loadConfig().catch(showError);
  </script>
</body>
</html>"#;

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
    fn operator_index_fetches_config_endpoints() {
        assert!(OPERATOR_INDEX_HTML.contains("/operator/config/schema"));
        assert!(OPERATOR_INDEX_HTML.contains("/operator/config/resolved"));
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
