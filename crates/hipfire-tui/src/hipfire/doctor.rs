// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! In-TUI "doctor": runs `hipfire diag --json` on a background thread and parses
//! its diagnostics object into pass/fail checks the System tab renders.
//!
//! On-demand only (the live-GPU probe spawns the daemon, so it is slow) — never
//! per-frame. The single report arrives on an mpsc channel the App drains, the
//! same pattern as serve control. Honest: a missing field is a failed/!ok check
//! with a reason, never a fabricated pass.

use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use serde_json::Value;

use crate::hipfire::dashboard::{run_bounded, RocmSmiRun};
use crate::hipfire::native_cli_command;

/// Hard wall-clock bound on `hipfire diag` — it spawns the daemon (GPU init), so
/// it is much heavier than rocm-smi, but a hung daemon must not leak the worker
/// thread or an orphan holding VRAM forever.
const DOCTOR_TIMEOUT: Duration = Duration::from_secs(30);

/// One diagnostic line: a name, whether it passed, and a short detail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoctorCheck {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

/// Parsed `hipfire diag --json` result, or a transport-level `error`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DoctorReport {
    pub checks: Vec<DoctorCheck>,
    pub error: Option<String>,
}

/// Spawn native `hipfire diag --json` on a background thread; the single
/// report arrives on the returned receiver.
pub fn run() -> Receiver<DoctorReport> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(run_inner());
    });
    rx
}

fn run_inner() -> DoctorReport {
    let mut cmd = match native_cli_command() {
        Some(c) => c,
        None => {
            return DoctorReport {
                checks: Vec::new(),
                error: Some(
                    "native hipfire binary not found (set HIPFIRE_CLI_BIN or install hipfire)"
                        .into(),
                ),
            }
        }
    };
    cmd.arg("diag").arg("--json");
    let err = |msg: String| DoctorReport {
        checks: Vec::new(),
        error: Some(msg),
    };
    match run_bounded(cmd, DOCTOR_TIMEOUT) {
        RocmSmiRun::Output(stdout) => match extract_json_object(&stdout) {
            Some(json) => parse_diag_json(json),
            None => err("diag produced no JSON object".into()),
        },
        RocmSmiRun::NotFound => err("hipfire not found on PATH".into()),
        RocmSmiRun::TimedOut => err(format!(
            "diag timed out after {}s (killed)",
            DOCTOR_TIMEOUT.as_secs()
        )),
        RocmSmiRun::SpawnError(e) => err(format!("diag spawn failed: {e}")),
    }
}

/// Extract the top-level JSON object — first `{` to last `}` — from diag's
/// stdout, tolerating leading log lines AND pretty multi-line formatting (the
/// CLI emits `JSON.stringify(..., null, 2)`, so a single-line scan would miss it).
fn extract_json_object(stdout: &str) -> Option<&str> {
    let start = stdout.find('{')?;
    let end = stdout.rfind('}')?;
    (end >= start).then(|| &stdout[start..=end])
}

fn check(name: &str, ok: bool, detail: impl Into<String>) -> DoctorCheck {
    DoctorCheck {
        name: name.into(),
        ok,
        detail: detail.into(),
    }
}

/// Derive pass/fail checks from the `hipfire diag --json` object. Defensive: any
/// missing field becomes a failed check with an honest reason.
pub fn parse_diag_json(body: &str) -> DoctorReport {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return DoctorReport {
                checks: Vec::new(),
                error: Some(format!("diag JSON parse failed: {e}")),
            }
        }
    };

    let mut checks = Vec::new();

    // Platform — informational, always shown ok.
    checks.push(check(
        "platform",
        true,
        v["platform"].as_str().unwrap_or("unknown"),
    ));

    let amdgpu = v["amdgpu_loaded"].as_bool().unwrap_or(false);
    checks.push(check(
        "amdgpu module",
        amdgpu,
        if amdgpu { "loaded" } else { "not loaded" },
    ));

    let kfd = v["kfd"].as_bool().unwrap_or(false);
    checks.push(check(
        "/dev/kfd",
        kfd,
        if kfd { "present" } else { "missing" },
    ));

    let hipcc = v["rocm"]["hipcc"].as_str().filter(|s| !s.is_empty());
    checks.push(check(
        "hipcc (ROCm)",
        hipcc.is_some(),
        hipcc.unwrap_or("not found"),
    ));

    // Loose match: any non-empty string means a daemon was located (tolerates a
    // future "present"/path value); null/empty is the only failure.
    let daemon = v["daemon"].as_str().filter(|s| !s.is_empty());
    checks.push(check(
        "daemon binary",
        daemon.is_some(),
        daemon.unwrap_or("missing"),
    ));

    let n_gpus = v["gpus"].as_array().map(|a| a.len()).unwrap_or(0);
    checks.push(check("GPU (PCI)", n_gpus > 0, format!("{n_gpus} detected")));

    let n_models = v["models"].as_array().map(|a| a.len()).unwrap_or(0);
    checks.push(check(
        "local models",
        n_models > 0,
        format!("{n_models} present"),
    ));

    // Live GPU probe (daemon one-shot): ok iff an arch came back with no error.
    let gpu = &v["gpu"];
    let (gpu_ok, gpu_detail) = if let Some(s) = gpu.as_str() {
        // Non-object scalar (e.g. "disabled") — surface it, not a generic message.
        (false, s.to_string())
    } else {
        let gpu_err = gpu.get("error").and_then(Value::as_str);
        let gpu_arch = gpu.get("arch").and_then(Value::as_str);
        match (gpu_err, gpu_arch) {
            (Some(e), _) => (false, e.to_string()),
            (None, Some(arch)) => (true, arch.to_string()),
            (None, None) => (false, "no live probe".to_string()),
        }
    };
    checks.push(check("live GPU probe", gpu_ok, gpu_detail));

    DoctorReport {
        checks,
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_diag_extracts_pass_fail() {
        let body = r#"{
            "platform": "linux",
            "amdgpu_loaded": true,
            "kfd": true,
            "rocm": { "hipcc": "HIP version 6.2" },
            "daemon": "found",
            "gpus": ["AMD Radeon RX 7900 XTX"],
            "models": [{"name":"q","tag":"q","size":"5 GB"}],
            "gpu": { "arch": "gfx1100", "vram_total_mb": 24560 }
        }"#;
        let r = parse_diag_json(body);
        assert!(r.error.is_none());
        let by = |n: &str| r.checks.iter().find(|c| c.name == n).unwrap().ok;
        assert!(by("amdgpu module"));
        assert!(by("/dev/kfd"));
        assert!(by("hipcc (ROCm)"));
        assert!(by("daemon binary"));
        assert!(by("GPU (PCI)"));
        assert!(by("local models"));
        assert!(by("live GPU probe"));
    }

    #[test]
    fn parse_diag_marks_missing_as_failures() {
        // Honest: absent fields fail, the live probe surfaces its error.
        let body = r#"{
            "platform": "wsl2",
            "amdgpu_loaded": false,
            "kfd": false,
            "rocm": { "hipcc": null },
            "daemon": null,
            "gpus": [],
            "models": [],
            "gpu": { "error": "no device" }
        }"#;
        let r = parse_diag_json(body);
        let c = |n: &str| r.checks.iter().find(|c| c.name == n).unwrap();
        assert!(!c("amdgpu module").ok);
        assert!(!c("hipcc (ROCm)").ok);
        assert!(!c("daemon binary").ok);
        assert!(!c("GPU (PCI)").ok);
        assert!(!c("live GPU probe").ok);
        assert_eq!(c("live GPU probe").detail, "no device");
        assert!(c("platform").ok, "platform is informational");
    }

    #[test]
    fn parse_diag_bad_json_is_error() {
        let r = parse_diag_json("not json");
        assert!(r.error.is_some());
        assert!(r.checks.is_empty());
    }

    #[test]
    fn extract_json_object_handles_pretty_multiline_with_log_prefix() {
        // The CLI emits JSON.stringify(.., null, 2) — pretty + possibly after logs.
        let stdout = "warming up\n{\n  \"platform\": \"linux\",\n  \"kfd\": true\n}\n";
        let json = extract_json_object(stdout).expect("object extracted");
        let r = parse_diag_json(json);
        assert!(r.error.is_none(), "pretty multi-line JSON parses");
        assert!(r.checks.iter().any(|c| c.name == "/dev/kfd" && c.ok));
        // No object at all -> None.
        assert!(extract_json_object("no braces here").is_none());
    }

    #[test]
    fn parse_diag_daemon_loose_and_gpu_scalar() {
        // A non-"found" daemon string still counts as located; a scalar gpu is
        // surfaced verbatim rather than a generic message.
        let body = r#"{"daemon": "/home/u/.hipfire/bin/daemon", "gpu": "disabled"}"#;
        let r = parse_diag_json(body);
        let c = |n: &str| r.checks.iter().find(|c| c.name == n).unwrap();
        assert!(
            c("daemon binary").ok,
            "any non-empty daemon string is found"
        );
        assert!(!c("live GPU probe").ok);
        assert_eq!(c("live GPU probe").detail, "disabled");
    }
}
