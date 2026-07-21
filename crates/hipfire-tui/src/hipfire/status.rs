// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{fs, process::Stdio};

use anyhow::{anyhow, Result};

use super::{dashboard::Dashboard, HipfirePaths};

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
    /// Load ONLY the fast local state — pure file I/O (serve.pid read + `~/.hipfire`
    /// path-existence checks). Performs NO network (/health) or hardware (lspci)
    /// probe, so it is safe to call synchronously on the UI thread (startup and
    /// the `r` reload). The live fields (`serve_http_ok`, `health_text`,
    /// `gpu_lines`) start empty and are filled in by [`StatusState::overlay_live`]
    /// from the background `DashboardWorker` snapshot.
    pub fn load_local(paths: &HipfirePaths) -> Self {
        let serve_pid = fs::read_to_string(&paths.serve_pid)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok());
        let serve_pid_alive = serve_pid
            .map(|pid| std::path::Path::new(&format!("/proc/{pid}")).exists())
            .unwrap_or(false);
        let paths_ok = vec![
            ("~/.hipfire".into(), paths.root.exists()),
            ("models".into(), paths.models.exists()),
            ("config.toml".into(), paths.config.exists()),
            ("legacy config.json".into(), paths.legacy_config.exists()),
            ("models.toml".into(), paths.models_catalog.exists()),
            (
                "legacy models.json".into(),
                paths.legacy_models_catalog.exists(),
            ),
            (
                "legacy per_model_config.json".into(),
                paths.legacy_per_model_config.exists(),
            ),
            ("serve.log".into(), paths.serve_log.exists()),
        ];
        Self {
            serve_pid,
            serve_pid_alive,
            serve_http_ok: false,
            health_text: String::new(),
            gpu_lines: Vec::new(),
            paths_ok,
        }
    }

    /// Fold the latest background `DashboardWorker` snapshot into the live
    /// fields. This is the ONLY path through which `serve_http_ok` / `health_text`
    /// / `gpu_lines` are populated — the worker did the /health + rocm-smi probes
    /// OFF the UI thread, so this is a cheap in-memory copy with no I/O. Called
    /// every frame from `App::sync_dashboard`.
    pub fn overlay_live(&mut self, dash: &Dashboard) {
        self.serve_http_ok = dash.serve_up;
        self.health_text = dash.health_text.clone();
        self.gpu_lines = dash
            .system
            .as_ref()
            .map(gpu_lines_from_system)
            .unwrap_or_default();
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
    let mut cmd = super::native_cli_command().ok_or_else(|| {
        anyhow!("native hipfire binary not found (set HIPFIRE_CLI_BIN or install hipfire)")
    })?;
    cmd.arg("serve")
        .arg("-d")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| anyhow!("failed to launch `hipfire serve -d`: {err}"))?;
    Ok(())
}

/// Derive the Home-tab GPU summary lines from the background-probed
/// [`SystemInfo`] (rocm-smi product name + gfx arch) rather than running `lspci`
/// synchronously on the UI thread. The worker probed this OFF the UI thread, so
/// this is a pure in-memory format. Falls back to an honest hint when neither
/// field probed cleanly.
fn gpu_lines_from_system(system: &super::dashboard::SystemInfo) -> Vec<String> {
    let mut lines = Vec::new();
    if system.gpu_name.is_available() {
        lines.push(system.gpu_name.display().to_string());
    }
    if system.gpu_arch.is_available() {
        lines.push(format!("arch  {}", system.gpu_arch.display()));
    }
    if lines.is_empty() {
        lines.push("No GPU detected via rocm-smi. Run hipfire diag for full probe.".into());
    }
    lines
}
