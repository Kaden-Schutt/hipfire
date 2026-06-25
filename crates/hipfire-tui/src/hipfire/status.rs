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
    /// Live PIDs of `hipfire-serve` processes (scanned from /proc).
    pub serve_pids: Vec<u32>,
    /// Live PIDs of `hipfire-daemon` processes (daemon.pid + /proc scan).
    pub daemon_pids: Vec<u32>,
    /// Full `/v1/` endpoint URLs, including any `/etc/hosts` IP forms.
    pub endpoints: Vec<String>,
    /// Local system hostname for the title bar.
    pub hostname: String,
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
        let lock_lines = resource_lock_lines(&hipfire_lock::resource_lock_root());
        let log_lines = log_tail_lines(paths, 160);

        // hipfire-serve PIDs: prefer the recorded serve.pid, then any matching
        // /proc comm so multiple workers are surfaced.
        let mut serve_pids = pids_by_comm("hipfire-serve");
        if let Some(pid) = serve_pid {
            if !serve_pids.contains(&pid) && Path::new(&format!("/proc/{pid}")).exists() {
                serve_pids.push(pid);
            }
        }
        serve_pids.sort_unstable();
        serve_pids.dedup();

        // hipfire-daemon PIDs: the singleton records itself in daemon.pid; also
        // scan /proc in case the lock file is stale or missing.
        let mut daemon_pids = pids_by_comm("hipfire-daemon");
        if let Some(pid) = fs::read_to_string(&paths.daemon_pid)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
        {
            if !daemon_pids.contains(&pid) && Path::new(&format!("/proc/{pid}")).exists() {
                daemon_pids.push(pid);
            }
        }
        daemon_pids.sort_unstable();
        daemon_pids.dedup();

        let hostname = system_hostname();
        let endpoints = endpoint_urls(config, &hostname);

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
            serve_pids,
            daemon_pids,
            endpoints,
            hostname,
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

/// Read the local system hostname (`/etc/hostname`, then `$HOSTNAME`).
fn system_hostname() -> String {
    if let Ok(raw) = fs::read_to_string("/etc/hostname") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    std::env::var("HOSTNAME").unwrap_or_else(|_| "localhost".into())
}

/// Scan `/proc/<pid>/comm` for processes whose command name matches `target`.
/// Linux truncates comm to 15 bytes; `hipfire-serve`/`hipfire-daemon` both fit.
fn pids_by_comm(target: &str) -> Vec<u32> {
    let mut pids = Vec::new();
    let Ok(entries) = fs::read_dir("/proc") else {
        return pids;
    };
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        let comm = fs::read_to_string(format!("/proc/{pid}/comm")).unwrap_or_default();
        if comm.trim() == target {
            pids.push(pid);
        }
    }
    pids.sort_unstable();
    pids
}

/// Build the set of `/v1/` endpoint URLs a client could hit: the configured
/// reachable host, plus any `/etc/hosts` IP that maps to this hostname.
fn endpoint_urls(config: &ConfigState, hostname: &str) -> Vec<String> {
    let port = config.port;
    let mut urls = vec![format!("http://{}:{}/v1/", config.probe_host(), port)];
    // Reverse-map the local hostname through /etc/hosts so the IP form is shown
    // alongside the name form (no std reverse-DNS; parse the file directly).
    for (name, ip) in parse_etc_hosts() {
        if name == hostname && !ip.starts_with("127.") && ip != "::1" {
            urls.push(format!("http://{ip}:{port}/v1/"));
            urls.push(format!("http://{name}:{port}/v1/"));
        }
    }
    urls.sort();
    urls.dedup();
    urls
}

/// Parse `/etc/hosts` into (name, ip) pairs. Best-effort; comments stripped.
fn parse_etc_hosts() -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Ok(text) = fs::read_to_string("/etc/hosts") else {
        return out;
    };
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split_whitespace();
        let Some(ip) = fields.next() else { continue };
        for name in fields {
            out.push((name.to_string(), ip.to_string()));
        }
    }
    out
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

/// Send SIGTERM to the given PIDs (graceful stop). Returns how many signals
/// were dispatched without error.
pub fn stop_pids(pids: &[u32]) -> Result<usize> {
    if pids.is_empty() {
        return Err(anyhow!("no running process to stop"));
    }
    let mut stopped = 0;
    for pid in pids {
        let ok = Command::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            stopped += 1;
        }
    }
    Ok(stopped)
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
    // flock(2)-based leases: probe the live kernel lock state for the shared GPU
    // lock (`hipfire_lock::gpu_resource_lock_path`) + any per-resource flock files under
    // lock_dir. A lockfile existing is NOT "held" — only the kernel flock is.
    use hipfire_lock::{gpu_resource_lock_path, probe, LockState};
    use std::collections::BTreeSet;
    let mut targets: Vec<(String, std::path::PathBuf)> = Vec::new();
    let mut seen = BTreeSet::new();
    let gpu = gpu_resource_lock_path();
    if gpu.exists() {
        seen.insert(gpu.clone());
        targets.push(("hip-gpu-0".to_string(), gpu));
    }
    if let Ok(entries) = fs::read_dir(lock_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path.extension().and_then(|x| x.to_str()) == Some("lock")
                && seen.insert(path.clone())
            {
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                targets.push((name, path));
            }
        }
    }
    let mut lines: Vec<String> = targets
        .into_iter()
        .map(|(name, path)| match probe(&path) {
            Ok(LockState::Busy(holder)) => {
                let h = if holder.is_empty() { "held" } else { &holder };
                format!("{name}: BUSY {h}")
            }
            Ok(LockState::Free) => format!("{name}: free"),
            Err(_) => format!("{name}: (probe failed)"),
        })
        .collect();
    lines.sort();
    if lines.is_empty() {
        lines.push("No resource locks held".to_string());
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
