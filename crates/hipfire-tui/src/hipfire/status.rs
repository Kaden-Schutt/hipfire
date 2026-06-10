// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    env, fs,
    process::{Command, Stdio},
    time::Duration,
};

use anyhow::{anyhow, Result};

use super::{config::ConfigState, HipfirePaths};

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
        let serve_pid = fs::read_to_string(&paths.serve_pid)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok());
        let serve_pid_alive = serve_pid
            .map(|pid| std::path::Path::new(&format!("/proc/{pid}")).exists())
            .unwrap_or(false);
        let (serve_http_ok, health_text) = probe_health(config);
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

pub fn start_background_serve() -> Result<()> {
    let cwd = env::current_dir()?;
    let script = cwd.join("cli/index.ts");
    if !script.exists() {
        return Err(anyhow!(
            "cli/index.ts not found; run this spike from the hipfire repo root"
        ));
    }

    Command::new("bun")
        .arg(script)
        .arg("serve")
        .arg("-d")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| anyhow!("failed to launch `bun cli/index.ts serve -d`: {err}"))?;
    Ok(())
}

fn probe_health(config: &ConfigState) -> (bool, String) {
    let url = format!("http://{}:{}/health", config.probe_host(), config.port);
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
