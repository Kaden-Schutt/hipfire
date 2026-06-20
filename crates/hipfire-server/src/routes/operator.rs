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
    .tabs {
      display: flex;
      gap: 8px;
      padding-bottom: 16px;
    }
    .tab {
      min-width: 96px;
    }
    .tab.active {
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 700;
    }
    .panel[hidden] { display: none; }
    .grid {
      display: grid;
      grid-template-columns: minmax(260px, 0.9fr) minmax(340px, 1.1fr);
      gap: 16px;
      align-items: start;
    }
    .section-title {
      margin: 0 0 10px;
      font-size: 14px;
      color: var(--muted);
      font-weight: 700;
    }
    .event-list {
      display: grid;
      gap: 8px;
    }
    .event {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 9px 10px;
      overflow-wrap: anywhere;
    }
    .event strong {
      color: var(--accent);
      font-size: 12px;
    }
    .event code {
      display: block;
      margin-top: 5px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: var(--muted);
      white-space: pre-wrap;
    }
    tr.selectable { cursor: pointer; }
    tr.selected { background: color-mix(in srgb, var(--accent) 14%, transparent); }
    @media (max-width: 820px) {
      header, main { padding-left: 14px; padding-right: 14px; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
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
    <nav class="tabs" aria-label="Operator sections">
      <button class="tab active" id="tab-config" type="button">Config</button>
      <button class="tab" id="tab-training" type="button">Training</button>
    </nav>
    <section id="config-panel" class="panel">
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
    </section>
    <section id="training-panel" class="panel" hidden>
      <div class="toolbar">
        <button id="training-refresh" type="button">Refresh</button>
      </div>
      <section class="summary" aria-label="Training summary">
        <div class="metric"><span>Runs</span><strong id="training-count">-</strong></div>
        <div class="metric"><span>Active</span><strong id="training-active">-</strong></div>
        <div class="metric"><span>Stale</span><strong id="training-stale">-</strong></div>
        <div class="metric"><span>Directory</span><strong id="training-dir">-</strong></div>
      </section>
      <div class="grid">
        <section>
          <h2 class="section-title">Runs</h2>
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Status</th>
                <th>Phase</th>
                <th>Progress</th>
                <th>Best</th>
              </tr>
            </thead>
            <tbody id="training-rows"></tbody>
          </table>
        </section>
        <section>
          <h2 class="section-title">Selected Run</h2>
          <section class="summary" aria-label="Selected training run">
            <div class="metric"><span>Target</span><strong id="training-target">-</strong></div>
            <div class="metric"><span>Artifact</span><strong id="training-artifact">-</strong></div>
            <div class="metric"><span>Checkpoint</span><strong id="training-checkpoint">-</strong></div>
            <div class="metric"><span>Admission</span><strong id="training-admission">-</strong></div>
          </section>
          <div id="training-events" class="event-list"></div>
        </section>
      </div>
    </section>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const tabConfigEl = document.getElementById("tab-config");
    const tabTrainingEl = document.getElementById("tab-training");
    const configPanelEl = document.getElementById("config-panel");
    const trainingPanelEl = document.getElementById("training-panel");
    const modelEl = document.getElementById("model");
    const refreshEl = document.getElementById("refresh");
    const rowsEl = document.getElementById("rows");
    const sourceEl = document.getElementById("source");
    const pathEl = document.getElementById("path");
    const fieldsEl = document.getElementById("fields");
    const diagnosticsEl = document.getElementById("diagnostics");
    const trainingRefreshEl = document.getElementById("training-refresh");
    const trainingCountEl = document.getElementById("training-count");
    const trainingActiveEl = document.getElementById("training-active");
    const trainingStaleEl = document.getElementById("training-stale");
    const trainingDirEl = document.getElementById("training-dir");
    const trainingRowsEl = document.getElementById("training-rows");
    const trainingTargetEl = document.getElementById("training-target");
    const trainingArtifactEl = document.getElementById("training-artifact");
    const trainingCheckpointEl = document.getElementById("training-checkpoint");
    const trainingAdmissionEl = document.getElementById("training-admission");
    const trainingEventsEl = document.getElementById("training-events");
    let selectedTrainingRun = null;

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

    async function loadTraining() {
      statusEl.textContent = "loading training";
      const resp = await fetch("/operator/training/runs");
      if (!resp.ok) throw new Error(`training ${resp.status}`);
      const payload = await resp.json();
      renderTraining(payload);
      const ids = (payload.runs || []).map((run) => run.id);
      const first = ids[0] || null;
      if (!selectedTrainingRun || !ids.includes(selectedTrainingRun)) selectedTrainingRun = first;
      if (selectedTrainingRun) {
        await loadTrainingDetail(selectedTrainingRun);
      } else {
        clearTrainingDetail();
      }
      statusEl.textContent = "training";
    }

    async function loadTrainingDetail(id) {
      selectedTrainingRun = id;
      const resp = await fetch(`/operator/training/runs/${encodeURIComponent(id)}`);
      if (!resp.ok) throw new Error(`training run ${resp.status}`);
      const detail = await resp.json();
      renderTrainingDetail(detail);
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

    function isActiveRun(run) {
      return ["queued", "capturing", "training", "evaluating", "checkpointing", "exporting"].includes(run.status || "");
    }

    function progressLabel(run) {
      const progress = run.progress || {};
      if (progress.percent !== undefined && progress.percent !== null) return `${Number(progress.percent).toFixed(1)}%`;
      if (progress.current_step !== undefined && progress.total_steps !== undefined) return `${progress.current_step}/${progress.total_steps}`;
      if (progress.current_step !== undefined) return String(progress.current_step);
      return "-";
    }

    function metricLabel(run) {
      const metrics = run.metrics || {};
      const value = metrics.best_eval_metric ?? metrics.eval_metric;
      return value === undefined || value === null ? "-" : Number(value).toFixed(4);
    }

    function renderTraining(payload) {
      const runs = payload.runs || [];
      trainingCountEl.textContent = String(runs.length);
      trainingActiveEl.textContent = String(runs.filter(isActiveRun).length);
      trainingStaleEl.textContent = String(runs.filter((run) => run.stale).length);
      trainingDirEl.textContent = payload.runs_dir || "-";
      trainingRowsEl.replaceChildren(...runs.map((run) => {
        const tr = document.createElement("tr");
        tr.className = `selectable${run.id === selectedTrainingRun ? " selected" : ""}`;
        tr.addEventListener("click", () => loadTrainingDetail(run.id).catch(showError));
        const id = document.createElement("td");
        id.className = "key";
        id.textContent = run.id || "-";
        const status = document.createElement("td");
        status.className = run.last_error || run.read_error ? "warn" : "";
        status.textContent = run.stale ? `${run.status || "unknown"} stale` : run.status || "unknown";
        const phase = document.createElement("td");
        phase.textContent = (run.progress && run.progress.phase) || run.status || "unknown";
        const progress = document.createElement("td");
        progress.textContent = progressLabel(run);
        const best = document.createElement("td");
        best.textContent = metricLabel(run);
        tr.append(id, status, phase, progress, best);
        return tr;
      }));
      if (!runs.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 5;
        cell.className = "muted";
        cell.textContent = "No training runs found.";
        row.append(cell);
        trainingRowsEl.replaceChildren(row);
      }
    }

    function renderTrainingDetail(detail) {
      const run = detail.summary || {};
      trainingTargetEl.textContent = run.target_model || "-";
      trainingArtifactEl.textContent = run.artifact || (run.handoff && run.handoff.artifact) || "-";
      trainingCheckpointEl.textContent = run.checkpoint && (run.checkpoint.path || run.checkpoint.state) || "-";
      trainingAdmissionEl.textContent = run.handoff && (run.handoff.admission_verdict || run.handoff.admission_status) || "-";
      const events = detail.recent_events || [];
      const errors = detail.event_errors || [];
      const cards = [];
      if (run.last_error || run.read_error) {
        const div = document.createElement("div");
        div.className = "event warn";
        div.innerHTML = `<strong>latest issue</strong><code></code>`;
        div.querySelector("code").textContent = (run.last_error && run.last_error.message) || run.read_error || "";
        cards.push(div);
      }
      for (const record of events.slice(-12).reverse()) {
        const div = document.createElement("div");
        div.className = "event";
        const event = record.event || {};
        const title = document.createElement("strong");
        title.textContent = `${record.line}: ${event.type || "unknown"}`;
        const code = document.createElement("code");
        code.textContent = JSON.stringify(event, null, 2);
        div.append(title, code);
        cards.push(div);
      }
      for (const err of errors.slice(-4)) {
        const div = document.createElement("div");
        div.className = "event warn";
        const title = document.createElement("strong");
        title.textContent = `line ${err.line}: malformed event`;
        const code = document.createElement("code");
        code.textContent = err.message;
        div.append(title, code);
        cards.push(div);
      }
      if (!cards.length) {
        const div = document.createElement("div");
        div.className = "event muted";
        div.textContent = "No events recorded for this run.";
        cards.push(div);
      }
      trainingEventsEl.replaceChildren(...cards);
      for (const row of trainingRowsEl.querySelectorAll("tr")) {
        const idCell = row.querySelector("td");
        row.classList.toggle("selected", idCell && idCell.textContent === selectedTrainingRun);
      }
    }

    function clearTrainingDetail() {
      trainingTargetEl.textContent = "-";
      trainingArtifactEl.textContent = "-";
      trainingCheckpointEl.textContent = "-";
      trainingAdmissionEl.textContent = "-";
      const div = document.createElement("div");
      div.className = "event muted";
      div.textContent = "No selected training run.";
      trainingEventsEl.replaceChildren(div);
    }

    function showTab(name) {
      const training = name === "training";
      configPanelEl.hidden = training;
      trainingPanelEl.hidden = !training;
      tabConfigEl.classList.toggle("active", !training);
      tabTrainingEl.classList.toggle("active", training);
      if (training) loadTraining().catch(showError);
    }

    refreshEl.addEventListener("click", () => loadConfig().catch(showError));
    trainingRefreshEl.addEventListener("click", () => loadTraining().catch(showError));
    tabConfigEl.addEventListener("click", () => showTab("config"));
    tabTrainingEl.addEventListener("click", () => showTab("training"));
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
    fn operator_index_exposes_training_surface() {
        assert!(OPERATOR_INDEX_HTML.contains("Training"));
        assert!(OPERATOR_INDEX_HTML.contains("/operator/training/runs"));
        assert!(OPERATOR_INDEX_HTML.contains("training-events"));
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
