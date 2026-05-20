// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

pub const ATLAS_SCHEMA: &str = "hipfire.kernel_atlas.v0";

/// A single row in the kernel atlas corpus. Mirrors the JSONL shape
/// emitted by `scripts/kernel_atlas.py` so this Rust crate and the
/// Python analysis layer share the same on-disk format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasRow {
    #[serde(default = "default_schema")]
    pub schema: String,
    #[serde(default)]
    pub phase: String,
    #[serde(default)]
    pub workload_kind: String,
    #[serde(default)]
    pub model_size: String,
    #[serde(default)]
    pub quant: String,
    #[serde(default)]
    pub shape_bucket: String,
    #[serde(default)]
    pub run_index: Option<u32>,
    #[serde(default)]
    pub metrics: BTreeMap<String, Value>,
    #[serde(default)]
    pub artifacts: BTreeMap<String, Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

fn default_schema() -> String {
    ATLAS_SCHEMA.to_string()
}

impl AtlasRow {
    pub fn new(phase: impl Into<String>, workload_kind: impl Into<String>) -> Self {
        Self {
            schema: default_schema(),
            phase: phase.into(),
            workload_kind: workload_kind.into(),
            model_size: String::new(),
            quant: String::new(),
            shape_bucket: String::new(),
            run_index: None,
            metrics: BTreeMap::new(),
            artifacts: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }

    /// Insert a numeric metric. `f64` is the canonical numeric type for
    /// throughput / latency / bandwidth measurements.
    pub fn set_metric_f64(&mut self, key: impl Into<String>, value: f64) -> &mut Self {
        self.metrics.insert(key.into(), Value::from(value));
        self
    }

    pub fn set_metric_u64(&mut self, key: impl Into<String>, value: u64) -> &mut Self {
        self.metrics.insert(key.into(), Value::from(value));
        self
    }

    pub fn set_metric_str(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.metrics.insert(key.into(), Value::String(value.into()));
        self
    }

    pub fn set_extra(&mut self, key: impl Into<String>, value: Value) -> &mut Self {
        self.extra.insert(key.into(), value);
        self
    }

    pub fn metric_f64(&self, key: &str) -> Option<f64> {
        self.metrics.get(key).and_then(Value::as_f64)
    }

    pub fn metric_u64(&self, key: &str) -> Option<u64> {
        self.metrics.get(key).and_then(Value::as_u64)
    }

    pub fn artifact_array(&self, key: &str) -> Option<&Vec<Value>> {
        self.artifacts.get(key).and_then(Value::as_array)
    }

    pub fn to_value(&self) -> serde_json::Result<Value> {
        serde_json::to_value(self)
    }

    /// Append this row to a JSONL file, creating it if it doesn't exist.
    /// The file is opened in append mode so concurrent writers from
    /// independent processes coexist without overwriting each other.
    pub fn append_to_jsonl(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let line = serde_json::to_string(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writeln!(file, "{line}")?;
        Ok(())
    }
}

/// Load all rows from a JSONL (or single-line JSON, or JSON array) file.
pub fn load_rows(path: impl AsRef<Path>) -> Result<Vec<AtlasRow>, String> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let trimmed = text.trim_start();
    if trimmed.starts_with('[') {
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
    } else if trimmed.starts_with('{') && text.lines().count() == 1 {
        let row: AtlasRow =
            serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
        Ok(vec![row])
    } else {
        let mut rows = Vec::new();
        for (idx, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let row: AtlasRow = serde_json::from_str(line)
                .map_err(|e| format!("parse {} line {}: {e}", path.display(), idx + 1))?;
            rows.push(row);
        }
        Ok(rows)
    }
}

pub fn load_row(path: impl AsRef<Path>, row_index: usize) -> Result<AtlasRow, String> {
    let rows = load_rows(path)?;
    rows.get(row_index)
        .cloned()
        .ok_or_else(|| format!("row index {row_index} out of range; rows={}", rows.len()))
}

pub fn value_object(pairs: impl IntoIterator<Item = (String, Value)>) -> Value {
    Value::Object(Map::from_iter(pairs))
}

/// Open a fresh JSONL file for writing, truncating any existing content.
/// Use this when starting a new collection run; subsequent rows append
/// via `AtlasRow::append_to_jsonl`.
pub fn truncate_jsonl(path: impl AsRef<Path>) -> std::io::Result<()> {
    let _ = File::create(path)?;
    Ok(())
}
