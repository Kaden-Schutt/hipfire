// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::task::TaskBundle;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub command: String,
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub schema: String,
    pub task_id: String,
    pub status: String,
    pub commands: Vec<CommandResult>,
}

pub fn eval_task_file(path: impl AsRef<Path>, cwd: Option<&str>) -> Result<EvalResult, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("read task {}: {e}", path.as_ref().display()))?;
    let task: TaskBundle = serde_json::from_str(&text)
        .map_err(|e| format!("parse task {}: {e}", path.as_ref().display()))?;
    eval_task(&task, cwd)
}

pub fn eval_task(task: &TaskBundle, cwd: Option<&str>) -> Result<EvalResult, String> {
    let mut commands = Vec::new();
    for command in task
        .correctness_commands
        .iter()
        .chain(task.eval_commands.iter())
    {
        commands.push(run_shell(command, cwd)?);
    }
    let pass = commands.iter().all(|result| result.status == 0);
    Ok(EvalResult {
        schema: "hipfire.kernel_atlas.eval.v0".to_string(),
        task_id: task.task_id.clone(),
        status: if pass { "pass" } else { "fail" }.to_string(),
        commands,
    })
}

fn run_shell(command: &str, cwd: Option<&str>) -> Result<CommandResult, String> {
    let mut cmd = Command::new("sh");
    cmd.arg("-lc").arg(command);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    let output = cmd
        .output()
        .map_err(|e| format!("run command {command:?}: {e}"))?;
    Ok(CommandResult {
        command: command.to_string(),
        status: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}
