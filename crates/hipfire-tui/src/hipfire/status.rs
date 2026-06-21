// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    fs,
    path::{Path, PathBuf},
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
    pub kernel_lines: Vec<String>,
    pub lock_lines: Vec<String>,
    pub log_lines: Vec<String>,
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
            ("config.local.json".into(), paths.host_config.exists()),
            (
                "per_model_config.json".into(),
                paths.per_model_config.exists(),
            ),
            ("serve.log".into(), paths.serve_log.exists()),
            ("logs".into(), paths.logs.exists()),
            ("kernels".into(), paths.kernels.exists()),
        ];
        let kernel_lines = kernel_cache_lines(&paths.kernels);
        let lock_lines = resource_lock_lines(Path::new("/tmp/hipfire-resource-locks"));
        let log_lines = log_tail_lines(paths, 160);
        Self {
            serve_pid,
            serve_pid_alive,
            serve_http_ok,
            health_text,
            gpu_lines,
            paths_ok,
            kernel_lines,
            lock_lines,
            log_lines,
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
    Command::new("hipfire")
        .arg("serve")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| anyhow!("failed to launch `hipfire serve`: {err}"))?;
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

fn kernel_cache_lines(kernel_root: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(kernel_root) else {
        return vec![format!(
            "No kernel cache directory at {}",
            kernel_root.display()
        )];
    };
    let mut lines = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let mut hsaco = 0;
        let mut hash = 0;
        if let Ok(files) = fs::read_dir(&path) {
            for file in files.flatten() {
                match file.path().extension().and_then(|ext| ext.to_str()) {
                    Some("hsaco") => hsaco += 1,
                    Some("hash") => hash += 1,
                    _ => {}
                }
            }
        }
        let arch = entry.file_name().to_string_lossy().to_string();
        let balance = if hsaco == hash {
            "balanced"
        } else {
            "mismatch"
        };
        lines.push(format!("{arch}: {hsaco} hsaco / {hash} hash ({balance})"));
    }
    lines.sort();
    if lines.is_empty() {
        lines.push(format!(
            "No architecture kernel caches under {}",
            kernel_root.display()
        ));
    }
    lines
}

fn resource_lock_lines(lock_dir: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(lock_dir) else {
        return vec![format!(
            "No resource lock directory at {}",
            lock_dir.display()
        )];
    };
    let mut lines = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_file() {
                return None;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let content = fs::read_to_string(&path)
                .unwrap_or_default()
                .lines()
                .take(3)
                .collect::<Vec<_>>()
                .join(" ");
            Some(format!("{name}: {content}"))
        })
        .collect::<Vec<_>>();
    lines.sort();
    if lines.is_empty() {
        lines.push(format!("No active lock files under {}", lock_dir.display()));
    }
    lines
}

fn log_tail_lines(paths: &HipfirePaths, count: usize) -> Vec<String> {
    let mut files = vec![paths.serve_log.clone()];
    if let Ok(entries) = fs::read_dir(&paths.logs) {
        let mut extra = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("log"))
            .collect::<Vec<PathBuf>>();
        extra.sort();
        files.extend(extra);
    }

    let mut lines = Vec::new();
    for path in files {
        if !path.is_file() {
            continue;
        }
        let tail = tail_file(&path, count.min(200));
        lines.push(format!("== {} ==", path.display()));
        lines.extend(tail.lines().map(str::to_string));
    }
    if lines.is_empty() {
        lines.push("No known hipfire log files found.".into());
    }
    lines
}

fn tail_file(path: &Path, count: usize) -> String {
    let Ok(raw) = fs::read_to_string(path) else {
        return String::new();
    };
    let mut selected = raw.lines().rev().take(count).collect::<Vec<_>>();
    selected.reverse();
    selected.join("\n")
}
