// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    fs,
    path::PathBuf,
    process::{Command, Stdio},
    sync::{
        mpsc::{self, Receiver},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use anyhow::{anyhow, Result};

use super::{
    cli::{resolve_cli, serve_detach_args},
    config::ConfigState,
    HipfirePaths,
};

#[derive(Clone, Debug)]
pub struct StatusState {
    pub serve_pid: Option<u32>,
    pub serve_pid_alive: bool,
    pub serve_http_ok: bool,
    pub health_text: String,
    pub gpu_lines: Vec<String>,
    pub paths_ok: Vec<(String, bool)>,
}

impl StatusState {
    pub fn load(paths: &HipfirePaths, config: &ConfigState) -> Self {
        let (serve_pid, serve_pid_alive) = read_serve_pid(&paths.serve_pid);
        let (serve_http_ok, health_text) = probe_health_at(&config.probe_host(), config.port);
        let gpu_lines = detect_gpu_lines();
        let paths_ok = vec![
            ("~/.hipfire".into(), paths.root.exists()),
            ("models".into(), paths.models.exists()),
            ("config.json".into(), paths.config.exists()),
            (
                "per_model_config.json".into(),
                paths.per_model_config.exists(),
            ),
            ("serve.log".into(), paths.serve_log.exists()),
        ];
        Self {
            serve_pid,
            serve_pid_alive,
            serve_http_ok,
            health_text,
            gpu_lines,
            paths_ok,
        }
    }

    pub fn serve_label(&self) -> String {
        if self.serve_http_ok {
            "online".into()
        } else if self.serve_pid_alive {
            "pid alive, HTTP not ready".into()
        } else if self.serve_pid.is_some() {
            "stale pid".into()
        } else {
            "offline".into()
        }
    }
}

/// Launch `serve -d` through whichever hipfire CLI is available: the
/// installed `hipfire` wrapper first, then `bun cli/index.ts` (cwd, then
/// ~/.hipfire). Overridable via HIPFIRE_TUI_CLI. Returns the label of the
/// CLI that was used so the status line can say which one started serve.
pub fn start_background_serve() -> Result<String> {
    let cli = resolve_cli().ok_or_else(|| {
        anyhow!(
            "no hipfire CLI found: install hipfire, run from a checkout, or set HIPFIRE_TUI_CLI"
        )
    })?;
    Command::new(&cli.program)
        .args(&cli.leading_args)
        .args(serve_detach_args())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| anyhow!("failed to launch `{} serve -d`: {err}", cli.label))?;
    Ok(cli.label)
}

fn probe_health_at(host: &str, port: u16) -> (bool, String) {
    let url = format!("http://{host}:{port}/health");
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_millis(450))
        .build();

    match agent.get(&url).call() {
        Ok(resp) => {
            let status = resp.status();
            let body = resp.into_string().unwrap_or_default();
            (status < 400, body)
        }
        Err(ureq::Error::Status(code, resp)) => {
            let body = resp.into_string().unwrap_or_default();
            (false, format!("HTTP {code}: {body}"))
        }
        Err(err) => (false, err.to_string()),
    }
}

fn read_serve_pid(pid_path: &std::path::Path) -> (Option<u32>, bool) {
    let pid = fs::read_to_string(pid_path)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok());
    let alive = pid
        .map(|pid| std::path::Path::new(&format!("/proc/{pid}")).exists())
        .unwrap_or(false);
    (pid, alive)
}

/// One health snapshot from the background poller.
#[derive(Clone, Debug)]
pub struct HealthUpdate {
    pub serve_http_ok: bool,
    pub health_text: String,
    pub serve_pid: Option<u32>,
    pub serve_pid_alive: bool,
}

/// Shared probe endpoint, updated by the app when host/port change.
pub type ProbeTarget = Arc<Mutex<(String, u16)>>;

const HEALTH_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Spawn the 2s background health poller. The thread exits when the app
/// drops the receiver. The first probe fires immediately so startup state
/// is fresh without pressing r.
pub fn spawn_health_poller(serve_pid_path: PathBuf, target: ProbeTarget) -> Receiver<HealthUpdate> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || loop {
        let (host, port) = {
            let guard = target.lock().unwrap_or_else(|err| err.into_inner());
            guard.clone()
        };
        let (serve_http_ok, health_text) = probe_health_at(&host, port);
        let (serve_pid, serve_pid_alive) = read_serve_pid(&serve_pid_path);
        if tx
            .send(HealthUpdate {
                serve_http_ok,
                health_text,
                serve_pid,
                serve_pid_alive,
            })
            .is_err()
        {
            break;
        }
        thread::sleep(HEALTH_POLL_INTERVAL);
    });
    rx
}

fn detect_gpu_lines() -> Vec<String> {
    let mut lines = Vec::new();
    if let Ok(out) = Command::new("lspci").output() {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            let lower = line.to_lowercase();
            if lower.contains("amd")
                || lower.contains("ati")
                || lower.contains("vga")
                || lower.contains("display")
                || lower.contains("3d controller")
            {
                lines.push(line.trim().to_string());
            }
            if lines.len() >= 6 {
                break;
            }
        }
    }
    if lines.is_empty() {
        lines.push("No GPU lines from lspci. Run hipfire diag for full probe.".into());
    }
    lines
}
