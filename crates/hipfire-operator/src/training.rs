// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{
    cmp::Ordering,
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

pub const TRAINING_RUNS_RELATIVE_DIR: &str = "training/runs";
pub const STATUS_FILE: &str = "status.json";
pub const EVENTS_FILE: &str = "events.jsonl";
pub const DEFAULT_STALE_AFTER_SECS: u64 = 15 * 60;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TrainingRunSummary {
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub owner: Option<String>,
    #[serde(default)]
    pub target_model: Option<String>,
    #[serde(default)]
    pub artifact: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub started_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub completed_at: Option<String>,
    #[serde(default)]
    pub progress: TrainingProgress,
    #[serde(default)]
    pub metrics: TrainingMetrics,
    #[serde(default)]
    pub checkpoint: Option<TrainingCheckpoint>,
    #[serde(default)]
    pub handoff: Option<TrainingHandoff>,
    #[serde(default)]
    pub last_error: Option<TrainingIssue>,
    #[serde(default)]
    pub stale: bool,
    #[serde(default)]
    pub stale_after_secs: Option<u64>,
    #[serde(default)]
    pub run_dir: Option<String>,
    #[serde(default)]
    pub read_error: Option<String>,
}

impl TrainingRunSummary {
    pub fn status_label(&self) -> &str {
        if self.status.is_empty() {
            "unknown"
        } else {
            &self.status
        }
    }

    pub fn phase_label(&self) -> &str {
        self.progress
            .phase
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| self.status_label())
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self.status_label(),
            "queued" | "capturing" | "training" | "evaluating" | "checkpointing" | "exporting"
        )
    }

    pub fn best_metric_label(&self) -> String {
        self.metrics
            .best_eval_metric
            .or(self.metrics.eval_metric)
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "-".to_string())
    }

    pub fn progress_label(&self) -> String {
        if let Some(percent) = self.progress.percent {
            format!("{percent:.1}%")
        } else if let (Some(current), Some(total)) =
            (self.progress.current_step, self.progress.total_steps)
        {
            format!("{current}/{total}")
        } else if let Some(current) = self.progress.current_step {
            current.to_string()
        } else {
            "-".to_string()
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TrainingProgress {
    #[serde(default)]
    pub phase: Option<String>,
    #[serde(default)]
    pub current_step: Option<u64>,
    #[serde(default)]
    pub total_steps: Option<u64>,
    #[serde(default)]
    pub percent: Option<f64>,
    #[serde(default)]
    pub eta_seconds: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TrainingMetrics {
    #[serde(default)]
    pub loss: Option<f64>,
    #[serde(default)]
    pub eval_metric: Option<f64>,
    #[serde(default)]
    pub best_eval_metric: Option<f64>,
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default)]
    pub throughput: Option<f64>,
    #[serde(default)]
    pub wall_time_seconds: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TrainingCheckpoint {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub age_seconds: Option<u64>,
    #[serde(default)]
    pub resume_source: Option<String>,
    #[serde(default)]
    pub state: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TrainingHandoff {
    #[serde(default)]
    pub artifact: Option<String>,
    #[serde(default)]
    pub admission_status: Option<String>,
    #[serde(default)]
    pub admission_verdict: Option<String>,
    #[serde(default)]
    pub evidence: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TrainingIssue {
    #[serde(default)]
    pub level: Option<String>,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub phase: Option<String>,
    #[serde(default)]
    pub event_type: Option<String>,
    #[serde(default)]
    pub line: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrainingEvent {
    #[serde(rename = "type", default = "unknown_event_kind")]
    pub kind: String,
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(flatten, default)]
    pub payload: Map<String, Value>,
}

impl TrainingEvent {
    pub fn label(&self) -> &str {
        if self.kind.is_empty() {
            "unknown"
        } else {
            &self.kind
        }
    }

    pub fn message(&self) -> Option<&str> {
        self.payload
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| self.payload.get("error").and_then(Value::as_str))
    }
}

impl Default for TrainingEvent {
    fn default() -> Self {
        Self {
            kind: unknown_event_kind(),
            timestamp: None,
            payload: Map::new(),
        }
    }
}

fn unknown_event_kind() -> String {
    "unknown".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrainingEventRecord {
    pub line: u64,
    pub byte_offset: u64,
    pub event: TrainingEvent,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrainingEventReadError {
    pub line: u64,
    pub byte_offset: u64,
    pub message: String,
    #[serde(default)]
    pub raw: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TrainingRunList {
    pub runs_dir: String,
    pub runs: Vec<TrainingRunSummary>,
    #[serde(default)]
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TrainingRunDetail {
    pub summary: TrainingRunSummary,
    #[serde(default)]
    pub recent_events: Vec<TrainingEventRecord>,
    #[serde(default)]
    pub event_errors: Vec<TrainingEventReadError>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TrainingRunEvents {
    pub run_id: String,
    pub events_path: String,
    #[serde(default)]
    pub events: Vec<TrainingEventRecord>,
    #[serde(default)]
    pub errors: Vec<TrainingEventReadError>,
}

pub fn training_runs_dir(hipfire_root: impl AsRef<Path>) -> PathBuf {
    hipfire_root.as_ref().join(TRAINING_RUNS_RELATIVE_DIR)
}

pub fn list_training_runs(runs_dir: impl AsRef<Path>) -> TrainingRunList {
    let runs_dir = runs_dir.as_ref();
    let mut list = TrainingRunList {
        runs_dir: runs_dir.display().to_string(),
        ..Default::default()
    };

    let entries = match fs::read_dir(runs_dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return list,
        Err(err) => {
            list.errors
                .push(format!("failed to read training runs dir: {err}"));
            return list;
        }
    };

    for entry in entries {
        match entry {
            Ok(entry) => {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let id = entry.file_name().to_string_lossy().to_string();
                list.runs.push(load_training_run_summary(&path, &id));
            }
            Err(err) => list
                .errors
                .push(format!("failed to read training run entry: {err}")),
        }
    }

    list.runs.sort_by(compare_runs);
    list
}

pub fn load_training_run_detail(
    runs_dir: impl AsRef<Path>,
    run_id: &str,
    event_limit: usize,
) -> Option<TrainingRunDetail> {
    let run_dir = runs_dir.as_ref().join(run_id);
    if !run_dir.is_dir() {
        return None;
    }
    let summary = load_training_run_summary(&run_dir, run_id);
    let events = read_training_run_events(&run_dir, run_id, Some(event_limit));
    Some(TrainingRunDetail {
        summary,
        recent_events: events.events,
        event_errors: events.errors,
    })
}

pub fn read_training_run_events(
    run_dir: impl AsRef<Path>,
    run_id: &str,
    limit: Option<usize>,
) -> TrainingRunEvents {
    let events_path = run_dir.as_ref().join(EVENTS_FILE);
    let mut result = TrainingRunEvents {
        run_id: run_id.to_string(),
        events_path: events_path.display().to_string(),
        ..Default::default()
    };
    let file = match fs::File::open(&events_path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return result,
        Err(err) => {
            result.errors.push(TrainingEventReadError {
                line: 0,
                byte_offset: 0,
                message: format!("failed to open events.jsonl: {err}"),
                raw: String::new(),
            });
            return result;
        }
    };

    let mut offset = 0u64;
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line_no = idx as u64 + 1;
        match line {
            Ok(raw) => {
                let byte_offset = offset;
                offset = offset.saturating_add(raw.len() as u64 + 1);
                if raw.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<TrainingEvent>(&raw) {
                    Ok(event) => result.events.push(TrainingEventRecord {
                        line: line_no,
                        byte_offset,
                        event,
                    }),
                    Err(err) => result.errors.push(TrainingEventReadError {
                        line: line_no,
                        byte_offset,
                        message: err.to_string(),
                        raw,
                    }),
                }
            }
            Err(err) => result.errors.push(TrainingEventReadError {
                line: line_no,
                byte_offset: offset,
                message: err.to_string(),
                raw: String::new(),
            }),
        }
    }

    if let Some(limit) = limit {
        let len = result.events.len();
        if len > limit {
            result.events = result.events.split_off(len - limit);
        }
    }

    result
}

fn load_training_run_summary(run_dir: &Path, id: &str) -> TrainingRunSummary {
    let status_path = run_dir.join(STATUS_FILE);
    let mut summary = match fs::read_to_string(&status_path) {
        Ok(raw) => match serde_json::from_str::<TrainingRunSummary>(&raw) {
            Ok(mut parsed) => {
                if parsed.id.is_empty() {
                    parsed.id = id.to_string();
                }
                parsed
            }
            Err(err) => TrainingRunSummary {
                id: id.to_string(),
                status: "unknown".to_string(),
                read_error: Some(format!("failed to parse status.json: {err}")),
                ..Default::default()
            },
        },
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => TrainingRunSummary {
            id: id.to_string(),
            status: "unknown".to_string(),
            read_error: Some("missing status.json".to_string()),
            ..Default::default()
        },
        Err(err) => TrainingRunSummary {
            id: id.to_string(),
            status: "unknown".to_string(),
            read_error: Some(format!("failed to read status.json: {err}")),
            ..Default::default()
        },
    };

    if summary.kind.is_empty() {
        summary.kind = "unknown".to_string();
    }
    if summary.status.is_empty() {
        summary.status = "unknown".to_string();
    }
    summary.run_dir = Some(run_dir.display().to_string());
    summary.stale_after_secs = summary.stale_after_secs.or(Some(DEFAULT_STALE_AFTER_SECS));
    summary.stale = summary.stale || infer_stale(run_dir, &summary);
    summary
}

fn infer_stale(run_dir: &Path, summary: &TrainingRunSummary) -> bool {
    if !summary.is_active() {
        return false;
    }
    let stale_after = summary.stale_after_secs.unwrap_or(DEFAULT_STALE_AFTER_SECS);
    let modified = fs::metadata(run_dir.join(STATUS_FILE))
        .and_then(|m| m.modified())
        .or_else(|_| fs::metadata(run_dir).and_then(|m| m.modified()))
        .ok();
    let Some(modified) = modified else {
        return false;
    };
    let Ok(age) = SystemTime::now().duration_since(modified) else {
        return false;
    };
    age.as_secs() > stale_after
}

fn compare_runs(a: &TrainingRunSummary, b: &TrainingRunSummary) -> Ordering {
    b.is_active()
        .cmp(&a.is_active())
        .then_with(|| b.updated_at.cmp(&a.updated_at))
        .then_with(|| b.started_at.cmp(&a.started_at))
        .then_with(|| b.created_at.cmp(&a.created_at))
        .then_with(|| a.id.cmp(&b.id))
}

#[allow(dead_code)]
fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct TempTree {
        path: PathBuf,
    }

    impl TempTree {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("hipfire-operator-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write_run(root: &Path, id: &str, status: &str, updated_at: &str) -> PathBuf {
        let dir = root.join(id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join(STATUS_FILE),
            format!(
                r#"{{
                  "id":"{id}",
                  "kind":"drafter",
                  "status":"{status}",
                  "owner":"standalone",
                  "target_model":"qwen3.5-0.8b-bf16",
                  "artifact":"qwen3.5-0.8b-ssm.drafter.hfq",
                  "updated_at":"{updated_at}",
                  "progress":{{"phase":"{status}","current_step":3,"total_steps":10}},
                  "metrics":{{"loss":1.2,"eval_metric":0.41,"best_eval_metric":0.52}},
                  "checkpoint":{{"path":"/tmp/checkpoint.hfq","state":"written"}},
                  "handoff":{{"artifact":"qwen3.5-0.8b-ssm.drafter.hfq","admission_status":"pending","evidence":[]}}
                }}"#
            ),
        )
        .unwrap();
        dir
    }

    #[test]
    fn list_runs_handles_empty_missing_and_sorts_active_first() {
        let tmp = TempTree::new("list");
        let missing = tmp.path.join("missing");
        let empty = list_training_runs(&missing);
        assert!(empty.runs.is_empty());
        assert!(empty.errors.is_empty());

        let active = write_run(&tmp.path, "active", "training", "2026-06-20T10:00:00Z");
        let completed = write_run(&tmp.path, "completed", "completed", "2026-06-20T12:00:00Z");
        let stale = write_run(&tmp.path, "stale", "training", "2026-06-20T09:00:00Z");
        let failed = write_run(&tmp.path, "failed", "failed", "2026-06-20T11:00:00Z");
        fs::write(
            stale.join(STATUS_FILE),
            r#"{"id":"stale","kind":"drafter","status":"training","stale":true,"updated_at":"2026-06-20T09:00:00Z"}"#,
        )
        .unwrap();
        fs::create_dir_all(tmp.path.join("bad-status")).unwrap();
        fs::write(tmp.path.join("bad-status").join(STATUS_FILE), "{not-json").unwrap();
        fs::write(active.join(EVENTS_FILE), "").unwrap();
        fs::write(completed.join(EVENTS_FILE), "").unwrap();
        fs::write(failed.join(EVENTS_FILE), "").unwrap();

        let list = list_training_runs(&tmp.path);
        assert_eq!(list.runs.len(), 5);
        assert_eq!(list.runs[0].id, "active");
        assert_eq!(list.runs[0].status, "training");
        assert_eq!(list.runs[1].id, "stale");
        assert!(list.runs[1].stale);
        assert!(list.runs.iter().any(|run| run.status == "failed"));
        let bad = list.runs.iter().find(|run| run.id == "bad-status").unwrap();
        assert!(bad.read_error.as_deref().unwrap().contains("parse"));
    }

    #[test]
    fn event_reader_preserves_unknown_events_and_malformed_lines() {
        let tmp = TempTree::new("events");
        let run_dir = write_run(&tmp.path, "run-a", "failed", "2026-06-20T10:00:00Z");
        let mut file = fs::File::create(run_dir.join(EVENTS_FILE)).unwrap();
        writeln!(
            file,
            r#"{{"type":"train_progress","epoch":1,"train_loss":1.5}}"#
        )
        .unwrap();
        writeln!(file, r#"{{"type":"future_event","custom":42}}"#).unwrap();
        writeln!(file, "{{bad-json").unwrap();
        writeln!(file, r#"{{"message":"missing type but still useful"}}"#).unwrap();

        let events = read_training_run_events(&run_dir, "run-a", None);
        assert_eq!(events.events.len(), 3);
        assert_eq!(events.errors.len(), 1);
        assert_eq!(events.events[0].event.kind, "train_progress");
        assert_eq!(events.events[1].event.kind, "future_event");
        assert_eq!(events.events[2].event.kind, "unknown");
        assert_eq!(events.errors[0].line, 3);
        assert!(!events.errors[0].message.is_empty());
        assert_eq!(events.errors[0].raw, "{bad-json");
    }

    #[test]
    fn detail_combines_summary_recent_events_and_errors() {
        let tmp = TempTree::new("detail");
        let run_dir = write_run(&tmp.path, "run-b", "completed", "2026-06-20T10:00:00Z");
        fs::write(
            run_dir.join(EVENTS_FILE),
            concat!(
                "{\"type\":\"run_started\"}\n",
                "{\"type\":\"train_progress\",\"epoch\":1}\n",
                "broken\n",
                "{\"type\":\"run_done\"}\n"
            ),
        )
        .unwrap();

        let detail = load_training_run_detail(&tmp.path, "run-b", 2).unwrap();
        assert_eq!(detail.summary.id, "run-b");
        assert_eq!(detail.recent_events.len(), 2);
        assert_eq!(detail.recent_events[0].event.kind, "train_progress");
        assert_eq!(detail.recent_events[1].event.kind, "run_done");
        assert_eq!(detail.event_errors.len(), 1);
    }
}
