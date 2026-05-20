// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::schema::AtlasRow;
use crate::suggest::suggestions_for_row;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBundle {
    pub schema: String,
    pub task_id: String,
    pub objective: String,
    pub source: Value,
    pub allowed_files: Vec<String>,
    pub correctness_commands: Vec<String>,
    pub eval_commands: Vec<String>,
    pub constraints: Vec<String>,
    pub created_unix_s: u64,
}

pub fn task_from_row(
    row: &AtlasRow,
    task_id: Option<String>,
    allowed_files: Vec<String>,
    correctness_commands: Vec<String>,
) -> TaskBundle {
    let suggestion = suggestions_for_row(row, 1).into_iter().next();
    let objective = suggestion
        .as_ref()
        .map(|s| s.title.clone())
        .unwrap_or_else(|| "Optimize profiled hipfire kernel under Atlas gates".to_string());
    TaskBundle {
        schema: "hipfire.kernel_atlas.task.v0".to_string(),
        task_id: task_id.unwrap_or_else(|| generated_id("atlas-task")),
        objective,
        source: row.to_value().unwrap_or_else(|_| json!({})),
        allowed_files,
        correctness_commands,
        eval_commands: Vec::new(),
        constraints: vec![
            "Do not change model semantics without a correctness gate.".to_string(),
            "Record benchmark command, output, git diff, and lineage.".to_string(),
            "Keep edits inside allowed_files when provided.".to_string(),
        ],
        created_unix_s: now_unix_s(),
    }
}

pub fn pytorch_task(
    name: String,
    op: String,
    input_shapes: Vec<String>,
    dtype: String,
    eval_command: String,
    task_id: Option<String>,
    allowed_files: Vec<String>,
) -> TaskBundle {
    TaskBundle {
        schema: "hipfire.kernel_atlas.task.v0".to_string(),
        task_id: task_id.unwrap_or_else(|| generated_id("atlas-pytorch")),
        objective: format!("Implement or tune HIP kernel for PyTorch op {op} ({name})"),
        source: json!({
            "kind": "pytorch_shape",
            "name": name,
            "op": op,
            "input_shapes": input_shapes,
            "dtype": dtype,
        }),
        allowed_files,
        correctness_commands: Vec::new(),
        eval_commands: vec![eval_command],
        constraints: vec![
            "Preserve numerical tolerance stated by the eval command.".to_string(),
            "Report target arch, shape, dtype, and measured speedup.".to_string(),
        ],
        created_unix_s: now_unix_s(),
    }
}

pub fn now_unix_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn generated_id(prefix: &str) -> String {
    format!("{prefix}-{}", now_unix_s())
}
