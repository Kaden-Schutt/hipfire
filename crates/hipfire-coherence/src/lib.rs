// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared coherence detector policy and report serialization helpers.

use hipfire_detect::{
    attractor::{AttractorFirst128, AttractorLast128, LongStateCollapse},
    eos_immediate::EosImmediate,
    ngram::{LoopGuardMirror, NgramDensity},
    report::{prompt_md5, Report, ReportHeader},
    special_leak::SpecialLeak,
    think::{ThinkEmpty, ThinkStall},
    timing::StepTimeSpike,
    toolcall::ToolcallShape,
    whitespace_only::WhitespaceOnly,
    DetectorBank, Event, Severity, Verdict,
};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct DetectorProfile {
    pub agentic: bool,
    pub stall_tokens: Option<usize>,
    pub detect_timing: bool,
}

impl DetectorProfile {
    pub fn default_for_prompt(prompt: &str, system: Option<&str>) -> Self {
        Self {
            agentic: decide_agentic(prompt, system),
            stall_tokens: None,
            detect_timing: false,
        }
    }

    pub fn long_state() -> Self {
        Self {
            agentic: false,
            stall_tokens: None,
            detect_timing: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoherenceRunConfig {
    pub model: String,
    pub prompt: String,
    pub prompt_label: String,
    pub system: Option<String>,
    pub tools: Option<Value>,
    pub assistant_prefix: Option<String>,
    pub force_jinja_chat: bool,
    pub max_tokens: usize,
    pub temperature: f64,
    pub repeat_penalty: Option<f64>,
    pub repeat_window: Option<usize>,
    pub max_seq: usize,
    pub state: Option<String>,
    pub profile: DetectorProfile,
}

#[derive(Debug, Clone)]
pub struct CoherenceRunOutput {
    pub report: Report,
    pub generated_text: String,
    pub token_ids: Vec<u32>,
    pub max_seq: usize,
    pub max_tokens: usize,
    pub temperature: f64,
    pub repeat_penalty: Option<f64>,
    pub repeat_window: Option<usize>,
    pub state: Option<String>,
    pub tools_present: bool,
    pub force_jinja_chat: bool,
}

impl CoherenceRunOutput {
    pub fn hard_fails(&self) -> usize {
        self.report.hard_fails
    }

    pub fn soft_warns(&self) -> usize {
        self.report.soft_warns
    }

    pub fn artifact_value(&self) -> Value {
        json!({
            "schema": 1,
            "kind": "coherence",
            "status": if self.hard_fails() > 0 { "fail" } else { "collected" },
            "report": self.report,
            "generated_text": self.generated_text,
            "token_ids": self.token_ids,
            "state": self.state,
            "tools_present": self.tools_present,
            "force_jinja_chat": self.force_jinja_chat,
            "max_seq": self.max_seq,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repeat_penalty": self.repeat_penalty,
            "repeat_window": self.repeat_window,
        })
    }
}

#[derive(Debug, Default)]
struct DoneStats {
    total_tokens: usize,
    _total_visible_bytes: usize,
    generated_text: String,
    token_ids: Vec<u32>,
    wall_ms: u64,
    ttft_ms: u64,
    daemon_prefill_ms: f64,
    daemon_prefill_tok_s: f64,
    daemon_decode_tok_s: f64,
    daemon_ttft_ms: f64,
    daemon_tok_s: f64,
}

struct DaemonChild {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
}

impl DaemonChild {
    fn close_stdin(&mut self) {
        self.stdin = None;
    }
}

pub fn run_coherence(config: &CoherenceRunConfig) -> Result<CoherenceRunOutput, String> {
    let daemon = hipfire_daemon_adapter::find_daemon_bin_or_error().map_err(|e| e.to_string())?;
    let mut session = CoherenceDaemonSession::load(&daemon, config)?;
    session.run(config)
}

pub struct CoherenceDaemonSession {
    child: DaemonChild,
    force_jinja_chat: bool,
}

impl CoherenceDaemonSession {
    pub fn load(daemon: &Path, config: &CoherenceRunConfig) -> Result<Self, String> {
        let mut child = spawn_daemon(daemon, config.force_jinja_chat)?;

        let mut params = serde_json::Map::new();
        params.insert("max_seq".to_string(), json!(config.max_seq));
        if let Some(state) = config.state.as_deref() {
            params.insert("state_quant".to_string(), json!(state));
        }
        let load = json!({
            "type": "load",
            "model": config.model,
            "params": Value::Object(params),
        });
        send(&mut child, &load)?;
        let loaded = recv_until(&mut child, |_| {})?;
        let ty = loaded.get("type").and_then(Value::as_str).unwrap_or("");
        if ty != "loaded" {
            shutdown_daemon(&mut child);
            return Err(format!("expected loaded, got {ty}"));
        }

        Ok(Self {
            child,
            force_jinja_chat: config.force_jinja_chat,
        })
    }

    pub fn run(&mut self, config: &CoherenceRunConfig) -> Result<CoherenceRunOutput, String> {
        if config.force_jinja_chat != self.force_jinja_chat {
            return Err(format!(
                "coherence session force_jinja_chat={} cannot run request force_jinja_chat={}",
                self.force_jinja_chat, config.force_jinja_chat
            ));
        }
        self.reset()?;

        let mut bank = build_detector_bank(&config.profile);
        let stats = drive_generate(&mut self.child, &mut bank, config)?;

        Ok(coherence_output_from_stats(config, bank, stats))
    }

    pub fn reset(&mut self) -> Result<(), String> {
        send(&mut self.child, &json!({ "type": "reset" }))?;
        let reset = recv_until(&mut self.child, |_| {})?;
        let ty = reset.get("type").and_then(Value::as_str).unwrap_or("");
        if ty == "reset" {
            Ok(())
        } else {
            Err(format!("expected reset, got {ty}"))
        }
    }
}

impl Drop for CoherenceDaemonSession {
    fn drop(&mut self) {
        shutdown_daemon(&mut self.child);
    }
}

fn coherence_output_from_stats(
    config: &CoherenceRunConfig,
    mut bank: DetectorBank,
    stats: DoneStats,
) -> CoherenceRunOutput {
    let finals = bank.finalize();
    let (arch, host) = arch_host();
    let tok_s = if stats.wall_ms > 0 {
        stats.total_tokens as f64 * 1000.0 / stats.wall_ms as f64
    } else {
        0.0
    };
    let gen_tok_s = if stats.wall_ms > stats.ttft_ms && stats.total_tokens > 0 {
        let gen_ms = stats.wall_ms - stats.ttft_ms;
        stats.total_tokens as f64 * 1000.0 / gen_ms as f64
    } else {
        tok_s
    };
    let combined_for_md5 = format!(
        "{}\n----\n{}",
        config.system.as_deref().unwrap_or(""),
        config.prompt
    );
    let header = ReportHeader {
        prompt_md5: prompt_md5(combined_for_md5.as_bytes()),
        prompt_label: config.prompt_label.clone(),
        model: config.model.clone(),
        arch,
        host,
        total_tokens: stats.total_tokens,
        tok_s,
        gen_tok_s,
        ttft_ms: stats.ttft_ms,
        daemon_prefill_ms: stats.daemon_prefill_ms,
        daemon_prefill_tok_s: stats.daemon_prefill_tok_s,
        daemon_decode_tok_s: stats.daemon_decode_tok_s,
        daemon_ttft_ms: stats.daemon_ttft_ms,
        daemon_tok_s: stats.daemon_tok_s,
    };
    CoherenceRunOutput {
        report: Report::new(header, finals),
        generated_text: stats.generated_text,
        token_ids: stats.token_ids,
        max_seq: config.max_seq,
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        repeat_penalty: config.repeat_penalty,
        repeat_window: config.repeat_window,
        state: config.state.clone(),
        tools_present: config.tools.is_some(),
        force_jinja_chat: config.force_jinja_chat,
    }
}

pub fn daemon_binary_available() -> bool {
    hipfire_daemon_adapter::find_daemon_bin().is_some()
}

pub fn build_detector_bank(profile: &DetectorProfile) -> DetectorBank {
    let mut bank = DetectorBank::new();
    bank.add(Box::new(AttractorFirst128::new()));
    bank.add(Box::new(AttractorLast128::new()));
    bank.add(Box::new(LongStateCollapse::new()));
    bank.add(Box::new(NgramDensity::new()));
    bank.add(Box::new(LoopGuardMirror::new()));
    bank.add(Box::new(ThinkEmpty::new()));
    if let Some(budget) = profile.stall_tokens {
        bank.add(Box::new(ThinkStall::new(budget)));
    }
    bank.add(Box::new(SpecialLeak::new()));
    if profile.agentic {
        bank.add(Box::new(ToolcallShape::new()));
    }
    bank.add(Box::new(EosImmediate::new()));
    bank.add(Box::new(WhitespaceOnly::new()));
    if profile.detect_timing {
        bank.add(Box::new(StepTimeSpike::new()));
    }
    bank
}

pub fn decide_agentic(prompt: &str, system: Option<&str>) -> bool {
    let combined = format!("{}\n{}", system.unwrap_or(""), prompt);
    let s = combined.to_ascii_lowercase();
    s.contains("<tool_call>")
        || (s.contains("\"name\"") && s.contains("\"arguments\""))
        || (s.contains("function") && s.contains("\"arguments\""))
}

pub fn detector_rows(report: &Report) -> Vec<Value> {
    report
        .rows
        .iter()
        .map(|row| {
            json!({
                "detector": row.name,
                "status": match &row.verdict {
                    Verdict::Ok => "pass",
                    Verdict::Skip { .. } => "skip",
                    Verdict::Fired { severity: Severity::Warn, .. } => "warn",
                    Verdict::Fired { severity: Severity::Fail, .. } => "fail",
                },
                "detail": match &row.verdict {
                    Verdict::Ok => None,
                    Verdict::Skip { reason } => Some(reason.clone()),
                    Verdict::Fired { detail, .. } => Some(detail.clone()),
                },
            })
        })
        .collect()
}

fn drive_generate(
    d: &mut DaemonChild,
    bank: &mut DetectorBank,
    config: &CoherenceRunConfig,
) -> Result<DoneStats, String> {
    let mut req = json!({
        "type": "generate",
        "id": "coherence-1",
        "prompt": config.prompt,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    });
    if let Some(repeat_penalty) = config.repeat_penalty {
        req.as_object_mut()
            .unwrap()
            .insert("repeat_penalty".to_string(), json!(repeat_penalty));
    }
    if let Some(repeat_window) = config.repeat_window {
        req.as_object_mut()
            .unwrap()
            .insert("repeat_window".to_string(), json!(repeat_window));
    }
    if let Some(system) = config.system.as_deref() {
        req.as_object_mut()
            .unwrap()
            .insert("system".to_string(), json!(system));
    }
    if let Some(tools) = config.tools.clone() {
        req.as_object_mut()
            .unwrap()
            .insert("tools".to_string(), tools);
    }
    if let Some(prefix) = config.assistant_prefix.as_deref() {
        req.as_object_mut()
            .unwrap()
            .insert("assistant_prefix".to_string(), json!(prefix));
    }
    send(d, &req)?;

    let t_start = Instant::now();
    let mut visible_bytes: usize = 0;
    let mut generated_text = String::new();
    let mut token_ids: Vec<u32> = Vec::new();
    let mut ttft_ms: Option<u64> = None;
    loop {
        let mut line = String::new();
        let n = d
            .stdout
            .read_line(&mut line)
            .map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            return Err("daemon closed stdout during generation".to_string());
        }
        let v: Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match v.get("type").and_then(Value::as_str).unwrap_or("") {
            "committed" => {
                let tok_id = v.get("tok_id").and_then(Value::as_u64).unwrap_or(0) as u32;
                let pos = v.get("pos").and_then(Value::as_u64).unwrap_or(0) as usize;
                let t_ms = v.get("t_ms").and_then(Value::as_u64).unwrap_or(0);
                token_ids.push(tok_id);
                let ev = Event::Committed { tok_id, pos, t_ms };
                let _ = bank.observe(&ev);
            }
            "token" => {
                let text = v.get("text").and_then(Value::as_str).unwrap_or("");
                let synthetic = v.get("synthetic").and_then(Value::as_bool).unwrap_or(false);
                let t_ms = t_start.elapsed().as_millis() as u64;
                if !synthetic {
                    visible_bytes += text.len();
                    generated_text.push_str(text);
                    if ttft_ms.is_none() {
                        ttft_ms = Some(t_ms);
                    }
                }
                let ev = Event::Token {
                    text,
                    t_ms,
                    synthetic,
                };
                let _ = bank.observe(&ev);
            }
            "done" => {
                let total_tokens = v.get("tokens").and_then(Value::as_u64).unwrap_or(0) as usize;
                let wall_ms = t_start.elapsed().as_millis() as u64;
                let ttft = ttft_ms.unwrap_or(wall_ms);
                let ev = Event::Done {
                    total_tokens,
                    total_visible_bytes: visible_bytes,
                    wall_ms,
                    ttft_ms: ttft,
                };
                let _ = bank.observe(&ev);
                return Ok(DoneStats {
                    total_tokens,
                    _total_visible_bytes: visible_bytes,
                    generated_text,
                    token_ids,
                    wall_ms,
                    ttft_ms: ttft,
                    daemon_prefill_ms: v.get("prefill_ms").and_then(Value::as_f64).unwrap_or(0.0),
                    daemon_prefill_tok_s: v
                        .get("prefill_tok_s")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0),
                    daemon_decode_tok_s: v
                        .get("decode_tok_s")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0),
                    daemon_ttft_ms: v.get("ttft_ms").and_then(Value::as_f64).unwrap_or(0.0),
                    daemon_tok_s: v.get("tok_s").and_then(Value::as_f64).unwrap_or(0.0),
                });
            }
            "error" => {
                let msg = v.get("message").and_then(Value::as_str).unwrap_or("?");
                return Err(format!("daemon error: {msg}"));
            }
            _ => {}
        }
    }
}

fn spawn_daemon(daemon: &Path, force_jinja_chat: bool) -> Result<DaemonChild, String> {
    let mut cmd = Command::new(daemon);
    cmd.env("HIPFIRE_EMIT_TOKEN_IDS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    if force_jinja_chat {
        cmd.env("HIPFIRE_JINJA_CHAT", "1");
    }
    let mut child = cmd.spawn().map_err(|e| format!("spawn daemon: {e}"))?;
    let stdin = child.stdin.take().ok_or("daemon stdin")?;
    let stdout = BufReader::new(child.stdout.take().ok_or("daemon stdout")?);
    Ok(DaemonChild {
        child,
        stdin: Some(stdin),
        stdout,
    })
}

fn send(d: &mut DaemonChild, msg: &Value) -> Result<(), String> {
    let stdin = d.stdin.as_mut().ok_or("daemon stdin already closed")?;
    let line = serde_json::to_string(msg).map_err(|e| format!("encode: {e}"))?;
    writeln!(stdin, "{line}").map_err(|e| format!("write daemon: {e}"))?;
    stdin.flush().map_err(|e| format!("flush daemon: {e}"))?;
    Ok(())
}

fn recv_until<F>(d: &mut DaemonChild, mut visitor: F) -> Result<Value, String>
where
    F: FnMut(&Value),
{
    let mut line = String::new();
    loop {
        line.clear();
        let n = d
            .stdout
            .read_line(&mut line)
            .map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            return Err("daemon closed stdout unexpectedly".to_string());
        }
        let v: Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        visitor(&v);
        match v.get("type").and_then(Value::as_str).unwrap_or("") {
            "loaded" | "unloaded" | "done" | "reset" | "error" => return Ok(v),
            _ => {}
        }
    }
}

fn shutdown_daemon(child: &mut DaemonChild) {
    let _ = send(child, &json!({ "type": "unload" }));
    let _ = recv_until(child, |_| {});
    child.close_stdin();
    let _ = child.child.wait();
}

fn arch_host() -> (String, String) {
    let arch = std::env::var("HIPFIRE_BASELINE_ARCH").unwrap_or_else(|_| {
        if let Ok(out) = Command::new("amdgpu-arch").output() {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout);
                let line = s.lines().next().unwrap_or("").trim();
                if !line.is_empty() {
                    return line.to_string();
                }
            }
        }
        "unknown".to_string()
    });
    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| {
        Command::new("hostname")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    });
    (arch, host)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_detect::report::ReportHeader;

    fn header() -> ReportHeader {
        ReportHeader {
            prompt_md5: "deadbeef".to_string(),
            prompt_label: "test".to_string(),
            model: "qwen3.5-9b-mq4.hfq".to_string(),
            arch: "gfx1100".to_string(),
            host: "host".to_string(),
            total_tokens: 16,
            tok_s: 10.0,
            gen_tok_s: 12.0,
            ttft_ms: 3,
            daemon_prefill_ms: 0.0,
            daemon_prefill_tok_s: 0.0,
            daemon_decode_tok_s: 0.0,
            daemon_ttft_ms: 0.0,
            daemon_tok_s: 0.0,
        }
    }

    #[test]
    fn agentic_detection_uses_prompt_or_system() {
        assert!(decide_agentic(
            "call <tool_call>{\"name\":\"read\",\"arguments\":{}}</tool_call>",
            None
        ));
        assert!(decide_agentic(
            "plain",
            Some("Use function calls with \"arguments\" objects")
        ));
        assert!(!decide_agentic("plain prompt", Some("plain system")));
    }

    #[test]
    fn detector_bank_respects_profile_toggles() {
        let plain = build_detector_bank(&DetectorProfile {
            agentic: false,
            stall_tokens: None,
            detect_timing: false,
        });
        let rich = build_detector_bank(&DetectorProfile {
            agentic: true,
            stall_tokens: Some(128),
            detect_timing: true,
        });
        assert!(rich.len() > plain.len());
    }

    #[test]
    fn detector_rows_match_runtime_artifact_shape() {
        let report = Report::new(
            header(),
            vec![
                ("clean", Verdict::Ok),
                ("optional", Verdict::skip("disabled")),
                ("soft", Verdict::warn("low confidence")),
                ("hard", Verdict::fail("loop detected")),
            ],
        );

        let rows = detector_rows(&report);
        assert_eq!(
            rows,
            vec![
                json!({"detector": "clean", "status": "pass", "detail": null}),
                json!({"detector": "optional", "status": "skip", "detail": "disabled"}),
                json!({"detector": "soft", "status": "warn", "detail": "low confidence"}),
                json!({"detector": "hard", "status": "fail", "detail": "loop detected"}),
            ]
        );
    }

    #[test]
    fn run_output_artifact_value_matches_runtime_schema() {
        let output = CoherenceRunOutput {
            report: Report::new(header(), vec![("clean", Verdict::Ok)]),
            generated_text: "Paris".to_string(),
            token_ids: vec![100, 200],
            max_seq: 4096,
            max_tokens: 32,
            temperature: 0.0,
            repeat_penalty: Some(1.05),
            repeat_window: Some(128),
            state: Some("mq4".to_string()),
            tools_present: true,
            force_jinja_chat: false,
        };

        let artifact = output.artifact_value();
        assert_eq!(artifact["schema"], json!(1));
        assert_eq!(artifact["kind"], json!("coherence"));
        assert_eq!(artifact["status"], json!("collected"));
        assert_eq!(artifact["generated_text"], json!("Paris"));
        assert_eq!(artifact["token_ids"], json!([100, 200]));
        assert_eq!(artifact["state"], json!("mq4"));
        assert_eq!(artifact["tools_present"], json!(true));
        assert_eq!(artifact["force_jinja_chat"], json!(false));
        assert_eq!(artifact["max_seq"], json!(4096));
        assert_eq!(artifact["max_tokens"], json!(32));
        assert_eq!(artifact["temperature"], json!(0.0));
        assert_eq!(artifact["repeat_penalty"], json!(1.05));
        assert_eq!(artifact["repeat_window"], json!(128));
    }
}
